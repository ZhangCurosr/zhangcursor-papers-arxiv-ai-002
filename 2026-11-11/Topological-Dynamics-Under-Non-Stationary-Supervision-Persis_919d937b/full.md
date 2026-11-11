# Topological Dynamics Under Non-Stationary Supervision: Persistent Homology of Neural Network Weight Evolution

Mitra Varun Gogulapati

Independent Researcher

mitravarun22@gmail.com

Abstract—We investigate how persistent topology of neuronweight point clouds evolves under stable and non-stationary supervision. An identical multilayer perceptron is trained on Fashion-MNIST with correct labels, a fixed random permutation of the training-label vector, or a fresh permutation every epoch. Five paired seeds share initialization and minibatch order, yielding 15 models trained for 100 epochs. Raw and row-normalized incoming-weight clouds are analyzed with H0 and H1 Vietoris– Rips persistence at 22 checkpoints. Correct-label models reach 89.39% mean true-label test accuracy; fixed-random models fit 67.94% of their assigned training targets without generalizing to the original task. Correct-label models have descriptively greater time-averaged normalized fc3 H1 persistence than either random-label condition, but the configured exact comparisons do not reach conventional significance after Holm correction. Early condition classification achieves 60.0% mean balanced accuracy using topology, 66.7% using conventional diagnostics, and 60.0% using both. Thus, no incremental predictive improvement is demonstrated in this setting. The archive contains all 3,300 configured topology cases. Interpretation is limited by five seeds, a single architecture and dataset, 282 capped H1 distance comparisons, and deferred bottleneck analysis.

Index Terms—persistent homology, neural network training dynamics, non-stationary supervision, random labels, topological data analysis

## I. INTRODUCTION

Ordinary supervised training assumes a fixed relationship between examples and targets. Relabeling, changing teachers, and online adaptation can instead modify the objective while optimization is in progress. Understanding such dynamics requires separating target informativeness from target stability. A correct-versus-changing-label comparison alone cannot distinguish the two: random targets remove useful image–label structure, and changing them adds non-stationarity.

Random-label memorization demonstrates that neural networks can fit mappings unrelated to input semantics, although its extent depends on model, optimization, and training budget [1], [2]. Work on plasticity further establishes that networks can become less able to adapt during continued learning [3], [4]. Our changing-random condition is an extreme supervision intervention, not a standard continual-learning benchmark; we do not measure later-task plasticity or retention. A fixedrandom control is essential because it retains an arbitrary but temporally stable assignment.

Loss and accuracy quantify fitting and generalization, but do not exhaustively describe internal organization. Persistent homology [5] supplies a multiscale summary of a chosen representation; persistence landscapes provide vector-valued summaries [6]. Applied to incoming-weight vectors, these tools may expose changes in neuron similarity. Equally, the summaries may recapitulate ordinary geometric or spectral changes. Demonstrating a visually different trajectory is therefore weaker than demonstrating useful information beyond a conventional baseline.

We ask four questions: RQ1, how do topological trajectories differ across correct, fixed-random, and changing-random supervision? RQ2, how do raw and normalized representations affect the comparison? RQ3, how do topological changes associate with optimization diagnostics? RQ4, do early topological features improve condition prediction over simpler measurements?

Our contributions are a five-seed paired benchmark, a reproducible 22-checkpoint analysis of five hidden layers, a complete accounting of numerical coverage, and a grouped evaluation of incremental predictive value. The empirical scope is intentionally narrow. The results include partial arbitrarylabel fitting, descriptive persistence differences, limited exacttest resolution, and no demonstrated classification gain from adding topology. Neither causality nor a topological phase transition is claimed.

## II. RELATED WORK

## A. What Object Is Being Analyzed?

Neural persistence [7] analyzes a weighted network graph using zero-dimensional persistence. Convolutional persistence studies topological descriptors of network weights and filters [8]. These precedents mean that computing persistence during training is not itself a new contribution. Our object is a Euclidean cloud whose points are complete incoming-weight vectors. Its filtration values are distances between rows, rather than individual connection magnitudes. In particular, H1 loops in this Rips complex are not literal cycles in the feed-forward network.

![](images/c7827c419e1245497c7bca474802085b7446fe00cd668235c86be182ef97455b.jpg)  
Fig. 1. Experimental design. A shared fitting/validation partition and paired initialization/order support three independent supervision runs per seed. The topology is computed on incoming-weight row clouds at 22 checkpoints, not on the complete parameter space or the loss landscape. The official test set is reserved for final evaluation.

Girrbach et al. [9] identify substantial weight-variance dependence in a graph-based neural-persistence measure. Their result does not establish the same limitation for our construction, but motivates explicit scale controls and baseline comparisons. Activation-based analyses, including topology of data representations across layers [10], study another object that depends on probe inputs. We do not measure activation topology here.

## B. Supervision, Geometry, and Evaluation

The random-label literature [1], [2] motivates an arbitraryassignment control, while plasticity studies [3], [4] motivate studying changing targets without equating that intervention with loss of plasticity. Spectral analyses [11] motivate our singular-value and effective-rank baselines. Functionpreserving reparameterizations can alter parameter geometry [12]; row normalization should therefore not be mistaken for invariance to all equivalent network parameterizations.

The contribution is the controlled three-condition comparison and its baseline evaluation, rather than a claim of a first general connection between topology and learning. We adopt seed-level replication and artifact transparency in the spirit of reproducibility recommendations [13]. Multiple epochs or intervals from one run do not create additional independent replications.

## III. EXPERIMENTAL METHODS

## A. Data, Network, and Optimization

Fashion-MNIST [14] contains ten classes of $2 8 ~ \times ~ 2 8$ grayscale images. A stratified split of the official training set produces 54,000 fitting and 6,000 validation examples, shared by all conditions and seeds. The partition seed is 314159. Fitting-image pixel mean and population standard deviation, after division by 255, normalize all splits. Validation images never enter fitting or label permutation. Images are flattened without augmentation. The official 10,000-image test set is evaluated at the end of each completed 100-epoch run and is not used for feature or hyperparameter selection.

TABLE I  
DATASET AND TRAINING CONFIGURATION.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Fitting / validation / test</td><td>54,000 / 6,000 / 10,000</td></tr><tr><td>Optimizer / loss</td><td>AdamW / cross-entropy</td></tr><tr><td>Learning rate / weight decay</td><td>Constant 0.001 / 0.01</td></tr><tr><td>Batch size / epochs</td><td>256 /  100</td></tr><tr><td>Gradient clipping</td><td>Global L2 limit 1.0</td></tr><tr><td>Paired seeds</td><td>42, 123, 456, 789, 2026</td></tr><tr><td>Conditions / trained models</td><td>Three / 15</td></tr><tr><td>Topology checkpoints</td><td>22, including initialization</td></tr><tr><td>Cloud size</td><td>128 in fc1–fc4; 64 in fc5</td></tr><tr><td>Representations / homology</td><td>Raw and normalized / H0 and H1</td></tr><tr><td>Compute</td><td>One NVIDIA Tesla T4; CPU topology</td></tr></table>

The architecture is

$$
7 8 4 \to 1 0 2 4 \to 5 1 2 \to 2 5 6 \to 1 2 8 \to 6 4 \to 1 0 .\tag{1}
$$

The five hidden linear layers, fc1–fc5, use ReLU; the output layer supplies raw logits. There is no dropout or batch normalization. Biases are trained and saved, but are not coordinates in the weight clouds. The implementation uses PyTorch [15] and AdamW [16]. Table I gives the fixed training configuration.

Online loss is the sample-weighted minibatch loss encountered during the epoch. Gradient norm is the sample-weighted mean of global minibatch L2 norms measured before clipping. Weight norm is the global L2 norm of all trainable parameters, including biases. Update norm is the global displacement between consecutive epoch parameter states. Assignedtarget training accuracy, original-label training accuracy, and original-label validation accuracy use the end-of-epoch model. These quantities answer different questions and are not conflated.

## B. Paired Conditions and Randomness

For each seed, one initialization is cloned into three models with independent AdamW states. Minibatch order is a deterministic function of seed and epoch and is identical across conditions. Model initialization, sample shuffling, and target assignment use separate random streams.

In the correct condition, the original fitting labels remain fixed. In fixed random, one permutation of the fitting-label vector is reused for all epochs. In changing random, a new permutation is generated at each epoch boundary and retained throughout that epoch. This shuffles labels across examples; it is not a permutation of the ten class identities. Each assignment preserves the exact class histogram, and a SHA-256 hash is logged. The archive confirms a single target hash per correct/fixed run and 100 distinct hashes per changing run. The fixed condition is a stable arbitrary-target control, not an assumption of successful full memorization.

## C. Point-Cloud Construction

For $W _ { \ell } \ \in \ \mathbb { R } ^ { N _ { \ell } \times D _ { \ell } }$ , a neuron contributes the row vector $w _ { i } \in \mathbb { R } ^ { D _ { \ell } }$ . Fixed sampled indices $I _ { \ell }$ define

$$
\begin{array} { l } { { \displaystyle X _ { \ell } ^ { \mathrm { r a w } } = \{ w _ { i } : i \in I _ { \ell } \} } , } \\ { { \displaystyle X _ { \ell } ^ { \mathrm { n o r m } } = \left\{ \frac { w _ { i } } { \operatorname* { m a x } ( \| w _ { i } \| _ { 2 } , 1 0 ^ { - 1 2 } ) } : i \in I _ { \ell } \right\} . } } \end{array}\tag{2}
$$

Indices are sampled without replacement using a NumPy generator seeded by $2 7 1 8 + N _ { \ell }$ and sorted. They are reused across all seeds, conditions, epochs, and representations. All 64 fc5 rows are retained. An audit of all 3,300 saved clouds found no zero-norm sampled rows, identical per-layer index subsets, and identical initialization clouds across paired conditions.

Raw clouds mix directional and radial changes; normalized clouds remove individual row magnitudes. Neither normalization nor diagram comparison removes all architectural, sampling, or parameterization dependence. Layers have different ambient dimensions and point counts. We do not pool them into a single cloud or interpret their scales as automatically equivalent.

## D. Persistence and Numerical Summaries

Ripser.py computes Vietoris–Rips persistence [17] from the Euclidean distance matrix with maximum homology dimension 1, default $\mathbb { F } _ { 2 }$ coefficients, and no finite filtration truncation. H2 is not part of the primary computational budget. Stability theory motivates persistence as a perturbation-aware representation [18], but does not by itself imply that a particular summary measures generalization or is invariant to network reparameterization.

For finite intervals $( b _ { j } , d _ { j } )$ with lifetime $a _ { j } = d _ { j } - b _ { j }$ , total persistence and entropy are

$$
\begin{array} { l } { { \mathrm { T P } ( D ) = \displaystyle \sum _ { j } a _ { j } , \qquad p _ { j } = \frac { a _ { j } } { \mathrm { T P } ( D ) } , } } \\ { { \mathrm { E n t } ( D ) = - \displaystyle \sum _ { j : a _ { j } > 0 } p _ { j } \log p _ { j } . } } \end{array}\tag{3}
$$

Empty diagrams or zero total persistence give entropy zero. We also save maximum/mean finite lifetime and positive-lifetime feature count. The single essential H0 interval is omitted from finite-lifetime summaries and matching distances, but retained in the H0 Betti curve. Betti counts are $\beta _ { k } ( r ) = \# \{ j : b _ { j } \leq$ $r < d _ { j } \}$ ; a diagram’s total feature count is not its Betti number at every scale. Landscapes [6] are the ordered persistence tent functions.

The exact checkpoint set is

$$
\begin{array} { c } { { \mathcal { E } = \{ 0 , 1 , \ldots , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 , 4 0 , 5 0 , 6 0 , } } \\ { { 7 0 , 8 0 , 9 0 , 1 0 0 \} , \qquad \left| \mathcal { E } \right| = 2 2 . } } \end{array}\tag{4}
$$

The recorded protocol note saying 23 checkpoints is a counting error; the configured schedule and all saved topology cases agree at 22. Betti curves use 128 grid values and landscapes use 32 values and three ranks. Normalized grids are fixed on [0, 2]; each raw layer uses [0, 1.001 max $d _ { i j } ]$ over all analyzed cases. Appendix D specifies which grid-dependent features may enter early prediction.

## E. Diagram Distances and Change Rates

Persim implements a finite-diagram 1-Wasserstein matching sum with Euclidean ground cost:

$$
W _ { 1 , 2 } ( D , D ^ { \prime } ) = \operatorname* { m i n } _ { \gamma } \sum _ { u \in \tilde { D } } \| u - \gamma ( u ) \| _ { 2 } ,\tag{5}
$$

where the diagrams are augmented by diagonal points to permit unmatched intervals. The diagonal cost of $( b , d )$ is $( d -$ $b ) / \sqrt { 2 }$ . Empty–empty distance is zero; an empty–nonempty distance is the sum of these diagonal costs. This is a matching sum, not a feature-count-normalized mean.

We compare successive sampled checkpoints, matched conditions within a seed, and seeds within a condition. A temporal rate is $r _ { k } = W _ { 1 , 2 } ( D _ { e _ { k - 1 } } , D _ { e _ { k } } ) / ( e _ { k } - e _ { k - 1 } )$ . It is an average change over the sampled interval, not a measurement at every intervening epoch. The saved evaluation contains 26,100 comparisons across homology dimensions, layers, representations, and comparison types. A comparison is skipped if either finite diagram exceeds 400 features. All 282 skips occur in normalized H1; they are missing values, never zeros. Bottleneck computations were deferred and no bottleneck values are reported.

## F. Optimization Associations and Early Prediction

The exploratory association analysis first divides adjacent checkpoint differences by elapsed epochs, then computes within-seed Spearman correlations between topologicalsummary rates and loss, gradient norm, update norm, validation accuracy, or weight norm. The saved topological summaries are H0 total persistence, H1 total persistence, and H1 entropy. Lag 1 shifts the topological-rate sequence by one sampled checkpoint; it is not a fixed one-epoch lag. These are not correlations of Wasserstein rate unless explicitly computed separately. No causal effect is estimated.

For early condition classification, one example per seed/condition is constructed by averaging features from epochs 1–5. The topology set contains finite H0/H1 summaries, normalized Betti areas and landscapes, and available temporal Wasserstein rates. Raw landscapes and raw Betti areas are excluded because their grid range uses later checkpoints. Normalized grids are fixed independently of later training. The baseline contains sampled-cloud row-norm and pairwise-distance statistics, three leading singular values and effective rank, plus loss, gradient norm, global weight norm, and update norm. Detailed feature accounting appears in Appendix H.

Scikit-learn [19] pipelines perform median imputation, standardization, and L2-regularized multinomial logistic regression. Nested leave-one-seed-out evaluation holds out all three models from one seed. The other four seeds supply 12 training examples; inner seed-held-out folds select $C \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ . All preprocessing is fitted inside the corresponding training fold. Outer balanced accuracy is the primary metric. Since every test seed has exactly one example per condition, a fold score is a multiple of $1 / 3$ . The outer training sets overlap, so comparisons across outer folds are descriptive rather than independent test replications.

## IV. SEED-LEVEL STATISTICAL ANALYSIS

The configured primary representation is normalized fc3, not a layer selected by searching the presented plots. For seed s and condition $c ,$ the temporal-average endpoint is

$$
A _ { s , c } = \frac { 1 } { 1 0 0 } \sum _ { k = 1 } ^ { 2 1 } \frac { \mathrm { T P } _ { s , c , e _ { k - 1 } } + \mathrm { T P } _ { s , c , e _ { k } } } { 2 } ( e _ { k } - e _ { k - 1 } ) .\tag{6}
$$

This is trapezoidal integration over the irregular schedule, not an unweighted average of 22 checkpoints. The second endpoint is the arithmetic mean of $r _ { k }$ over available normalized fc3 H1 temporal comparisons; unavailable intervals are omitted, making that endpoint conditional on coverage. The third comparison is topology minus baseline outer-fold balanced accuracy.

For each comparison, $\delta _ { s }$ denotes a paired seed difference. We report ${ \bar { \delta } } ,$ sample SD, $d _ { z } ~ = ~ \bar { \delta } / s _ { \delta }$ , and an unadjusted $t _ { 4 }$ interval $\bar { \delta } \pm t _ { 4 , 0 . 9 7 5 } s _ { \delta } / \sqrt { 5 }$ . The exact two-sided sign-flip test enumerates all $2 ^ { 5 }$ sign patterns and compares absolute mean differences. Its distributional interpretation assumes sign symmetry under the null; seeds, not epochs or intervals, are the units. The minimum attainable two-sided $p$ is $2 / 3 2 = 0 . 0 6 2 5$ preventing a conventional 0.05 rejection even before multiplicity adjustment.

Holm correction [20] includes all seven configured comparisons: three condition contrasts for each of the two topology endpoints and one topology-versus-baseline classification contrast. There is no claim of external preregistration. Table III reports the full family, including negative comparisons. Classification-fold tests are retained for transparency but are not assigned independent-replication status. The later visualization intervals, PCA, clustering, and association networks are exploratory. Different assumptions underlie an unadjusted t interval and an exact sign-flip test; exclusion of zero by the former is not a corrected significance result.

## V. RESULTS

## A. Coverage and Final Performance

All 15 configured models have contiguous 100-epoch histories and final test records. All $1 5 \times 2 2 \times 5 \times 2 = 3 3 0 0$ topology cases are present. The 26,100 comparison records comprise 6,300 temporal, 6,600 matched-condition, and 13,200 acrossseed evaluations. Of these, 25,818 have finite Wasserstein values. Appendices F and J distinguish complete case coverage from the deliberate distance skips.

Correct-label models achieve 99.51% assigned-target training accuracy and 89.39% true-label test accuracy on average (Table II). Fixed-random models achieve 67.94% against assigned training targets and 10.29% on the true test task. Thus the fixed-random control exhibits partial fitting, not complete arbitrary-label memorization. Changing-random models record 10.00% assigned-target and true-label test accuracy. This is an observed outcome of this schedule and optimizer, not a proof that learning within an epoch of changing targets is impossible. Figure 2 shows the accompanying optimization trajectories.

## B. Topological Trajectories and Layer Dependence

Figure 3 separates raw and normalized H1 trajectories. Raw curves mix magnitude and direction; normalization changes the relative appearance of conditions and layers. There is no single universal trajectory ordering. For example, final normalized H1 persistence can differ substantially between early and late layers, and fc5 uses half as many points as fc1– fc4. Complete final-checkpoint summaries appear in Table VI.

For the configured normalized fc3 endpoint, correct-label temporal-average H1 persistence exceeds fixed-random by 1.6156 and changing-random by 1.7836 units. The signs agree in all five paired seeds for both contrasts. However, each raw exact p is 0.0625 and each Holm-adjusted $p$ is 0.4375. Fixedrandom minus changing-random is 0.1680 units with adjusted $p = 1 . 0 0 0 0$ . These numerical differences are descriptive, not conventional significance claims.

Normalized H0 total persistence falls from initialization to epoch 100 in every layer/condition mean; the final magnitudes differ across conditions. This is a finite-interval mergingscale summary, not the number of components at every scale. Figure 4 instead displays actual live H1 counts as a function of filtration. Its concentration and scale shifts explain why one scalar total-persistence summary does not specify the full diagram geometry.

## C. Distances and Optimization Associations

Mean temporal-rate differences are smaller relative to seed variability than the total-persistence contrasts (Fig. 5). Fixedrandom minus changing-random is 0.0952, correct minus fixed-random is −0.0407, and correct minus changing-random is 0.0546; adjusted $p$ values are 0.9375 in all three cases. No claim that moving targets necessarily produce greater diagram displacement is supported by these comparisons.

All 282 capped distances are normalized H1 cases in fc1– fc4. Of these, 180 are recorded at epoch 0, 95 at epoch 1, and seven at epoch 2; temporal rows are indexed by their later endpoint. Missingness is therefore concentrated near initialization, not spread uniformly across training. This matters especially for early-change-rate features and comparisons conditional on available diagrams. H0 and raw H1 distances are fully available. Exact breakdowns are given in Tables XIII and XIV.

TABLE II  
FINAL PERFORMANCE ACROSS FIVE PAIRED SEEDS (PERCENT).
<table><tr><td>Condition</td><td></td><td>Assigned train True validation</td><td>True test</td></tr><tr><td>Correct</td><td> $9 9 . 5 1 \pm 0 . 2 1 $ </td><td> $8 9 . 7 8 \pm 0 . 1 9$ </td><td> $8 9 . 3 9 \pm 0 . 2 4$ </td></tr><tr><td>Fixed random</td><td> $6 7 . 9 4 \pm 4 . 8 9$ </td><td> $1 0 . 1 0 \pm 0 . 4 0$ </td><td> $1 0 . 2 9 \pm 0 . 6 1$ </td></tr><tr><td>Changing random</td><td> $1 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 . 0 0 \pm 0 . 0 0$ </td></tr></table>

Values are means ± one sample standard deviation across seeds. Assigned-target accuracy is evaluated using end-of-epoch weights. Test labels remain correct in every condition.

![](images/cc8867962980c3c1c56823b0744454d95c6f98f72c73a8c5ecbf4271383cad0a.jpg)

![](images/0e6b17248f033b355bc065ff2f77fa47ec59094d69bfec304f7f27d6e2411af9.jpg)

(c) Pre-clipping gradient norm  
![](images/c22cf88d20faa4fef4eb1e1e35a5683acf3dc356fc59e2df1919113061bf0fee.jpg)

(d) True-label validation accuracy  
![](images/0dfd79fc6aa29bb494c06ddd479b3f6f417d6ceddc5cfb96bbdfc511cf85b1c4.jpg)  
Fig. 2. Training behavior across five seeds. Lines are condition means; bands are ±1 sample SD across seeds, not confidence intervals. Correct-label models fit and generalize; fixed-random models progressively fit their assigned targets while remaining near chance on the original task; changing-random models remain near chance under the recorded configuration. Online loss and end-of-epoch accuracy have different measurement conventions.

TABLE III  
COMPLETE CONFIGURED SEVEN-COMPARISON FAMILY; n = 5 SEED-LEVEL DIFFERENCES.
<table><tr><td>Outcome</td><td>Comparison</td><td>Mean ∆</td><td>Unadjusted 95% t interval</td><td> $d _ { z }$ </td><td>Exact p</td><td>Holm p</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Fixed random — Changing random</td><td>0.1680</td><td>[-0.3488, 0.6849]</td><td>0.404</td><td>0.5000</td><td>1.0000</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Correct — Fixed random</td><td>1.6156</td><td>[1.3087, 1.9224]</td><td>6.538</td><td>0.0625</td><td>0.4375</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Correct — Changing random</td><td>1.7836</td><td>[1.0857, 2.4815]</td><td>3.173</td><td>0.0625</td><td>0.4375</td></tr><tr><td> ${ \overline { { r } } } _ { 1 }$ </td><td>Fixed random — Changing random</td><td>0.0952</td><td>[-0.0453, 0.2358]</td><td>0.841</td><td>0.1875</td><td>0.9375</td></tr><tr><td> ${ \overline { { r } } } _ { 1 }$ </td><td>Correct — Fixed random</td><td>-0.0407</td><td>[-0.1069,0.0256]</td><td>-0.762</td><td>0.1875</td><td>0.9375</td></tr><tr><td> ${ \overline { { r } } } _ { 1 }$ </td><td>Correct — Changing random</td><td>0.0546</td><td>[-0.0541, 0.1632]</td><td>0.624</td><td>0.2500</td><td>0.9375</td></tr><tr><td>∆BA</td><td>Topology – Baseline</td><td>-6.6667</td><td>[-41.2950,27.9617]</td><td>-0.239</td><td>1.0000</td><td>1.0000</td></tr></table>

TP<sub>1</sub>: time-averaged normalized fc3 H1 total persistence; ${ \overline { { r } } } _ { 1 } { : }$ mean available temporal Wasserstein rate; ∆BA: topology minus baseline, in percentage points. $d _ { z }$ is the mean difference divided by its sample SD. Classification-fold tests are descriptive because outer training sets overlap.

their visualization thresholds as tests.

The archived association analysis contains 4,500 withinseed feature/metric/lag correlations. Table VII reports the normalized fc3 H1 subset; remaining measurements are supplied as CSV. Changes are normalized by elapsed epochs, but common trends or causes may remain. We do not interpret the exploratory association networks as causal circuits or use

## D. Early Classification and Incremental Value

Table IV and Fig. 6 show every outer fold. Topology achieves 60.0% mean balanced accuracy, the baseline 66.7%, and the combined set 60.0%. Thus the tested topology features do not demonstrate improvement beyond the conventional diagnostics. The difference is one additional correct assignment among 15 held-out models, not compelling evidence that either family is generally superior. No confusion matrix is inferred because individual predicted labels were not retained.

![](images/7724ff7a7efafe3bfb7f467a82a2961c096a3983cf9dba6856aeabf794e92d6e.jpg)

![](images/a658a2a904db6af67d590ed11db5a13dc4ddc3bc6d30aa2dfb1b460f0657f1e3.jpg)

![](images/9506c23a6fc5f90df52e973c8253efb58e532b5007e3f030e0ad7f4aeba8748e.jpg)

![](images/70a6c27934e5703e568e7083942fbbb2dc12e6a62ae1c231c22153a0d09d6302.jpg)

![](images/7a8bcc80cbf8e4d8f251c3c9e67549f582e8ccc75991a10ea9138cf1319b5c14.jpg)

![](images/33121b04b64afd324aea16b4cf370101fccbe319fd09f4a7015dca32b6ab8762.jpg)  
Fig. 3. H1 total-persistence trajectories in representative early, middle, and late layers. Raw (top) and normalized (bottom) representations are kept separate Means and ±1 seed SD are shown; all 22 configured checkpoints are present. Connecting lines are guides between observations, not additional measured checkpoints. Different ambient dimensions and the 64-point fc5 cloud limit direct cross-layer interpretation.

![](images/8d58862344add6e93992dfbd1d8ba91dc59a145966bf87eb1bd644893e72e0b2.jpg)

![](images/76e8a8926b3099082edb2e1f9258902f2b2c49ec690212fb845162cc526b94db.jpg)

![](images/b256f70d24d6698e07c57eb2dbacc8d73378252deb7bc8e0ec8f567399c2346c.jpg)  
Fig. 4. Normalized fc3 H1 Betti surfaces reconstructed from saved finite intervals and averaged across five seeds. Columns use the same filtration window and nonlinear color normalization $( \gamma = 0 . 5 5 )$ to reveal low-count structure without changing the counts. Vertical positions enumerate the 22 observed checkpoints equally, not elapsed training time. The 160-value display grid is separate from the 128-value analysis grid. Betti counts are not total persistent-feature counts

The nominal input dimensionalities are 1,090 topology features, 94 baseline features, and 1,184 combined features, compared with 12 examples in an outer training fold. Regularization and nested selection limit direct leakage, but do not resolve this severe small-sample/high-dimensional setting or unequal feature budgets. The original descriptive topologyminus-baseline comparison is −6.67 percentage points, with exact and adjusted $p = 1 . 0 0 0 0 ;$ its broad unadjusted interval is reported in Table III.

## E. Expanded Descriptive Evidence

The appendix separates optimization, representation, and multivariate views rather than relying on a single topology trajectory. Tables VIII and IX give five training milestones for every condition, with seed variability retained. Table X extends the time-weighted H1 summary across all five hidden layers; Table XI pairs endpoint topology with geometric and spectral measurements. These additional summaries are exploratory and do not expand the primary multiplicity-adjusted testing family.

![](images/eb07994a952696635a174fadd1e1d3101bf66cc25a4626813f32c9e2248c319c.jpg)

![](images/588b05a62df7d6539cb3eaa1c0ed2bcf9f33b6d2b5dd984a55a39654d9d72a0c.jpg)  
Fig. 5. Configured normalized fc3 contrasts. Small points are five individual paired seed differences; diamonds and bars show the mean and unadjusted 95% $t _ { 4 }$ interval. Endpoint (a) is time-averaged H1 total persistence; (b) is mean available temporal Wasserstein rate. These intervals are not multiplicity-corrected and do not replace the exact/Holm results in Table III.

(a) Balanced accuracy (%)  
![](images/0a43501e64aed1619293805c56d872623c19280952f81c90db9e830e3da3feaa.jpg)

![](images/41cbb9e153d7cb469cb9ae6c443e37610ceff07a57a6a93ac665471a1d45efea.jpg)  
Fig. 6. Early condition classification with nested seed-held-out evaluation. (a) Actual fold-level balanced accuracies. (b) Paired baseline-relative differences in percentage points; overlapping topology/combined outcomes are offset vertically for visibility. Each fold has only three test examples, and outer training sets overlap. The plot retains the negative incremental-value result rather than implying that condition separation in full-trajectory figures establishes predictive usefulness.

TABLE IV  
NESTED LEAVE-ONE-SEED-OUT BALANCED ACCURACY (PERCENT).
<table><tr><td>Held-out seed</td><td>Topology</td><td>Baseline</td><td>Combined</td></tr><tr><td>42</td><td>66.67</td><td>33.33</td><td>66.67</td></tr><tr><td>123</td><td>66.67</td><td>100.00</td><td>66.67</td></tr><tr><td>456</td><td>66.67</td><td>66.67</td><td>66.67</td></tr><tr><td>789</td><td>33.33</td><td>66.67</td><td>33.33</td></tr><tr><td>2026</td><td>66.67</td><td>66.67</td><td>66.67</td></tr><tr><td>Mean</td><td>60.00</td><td>66.67</td><td>60.00</td></tr></table>

Each held-out seed contains three models, one per condition. The 6.67-point mean difference equals one additional correct assignment among 15 held-out models; individua predicted labels were not retained.

Quantitatively, normalized fc3 time-averaged H1 total persistence is $4 . 3 5 9 \pm 0 . 2 0 4$ under correct labels, $2 . 7 4 3 \pm 0 . 2 2 7$ under fixed random labels, and $2 . 5 7 5 \pm 0 . 4 0 8$ under changing random labels (mean ± sample SD across seeds). Raw fc3 values are $1 . 3 1 4 \pm 0 . 1 7 5 , 0 . 1 1 0 \pm 0 . 0 3 1$ , and $0 . 1 1 0 \pm 0 . 0 2 8 .$ respectively. Representation therefore changes both scale and the visible separation between randomized conditions. Depth also matters: normalized fc4 averages are $5 . 2 1 5 \pm 0 . 1 4 5$ $5 . 1 3 4 \pm 0 . 5 4 0$ , and $4 . 1 1 5 \pm 0 . 3 7 8$ , so the correct-versus-fixed separation is much smaller there than at fc3. These descriptive observations support reporting the full layer profile rather than generalizing one layer’s ordering to the entire network.

At epoch 50, fixed-random assigned-target accuracy is $3 0 . 1 2 7 \pm 8 . 1 8 4 \%$ and rises to 67.938 ± 4.887% by epoch 100, while true-label validation accuracy remains near chance. Correct-label validation accuracy is $8 9 . 8 3 0 { \pm } 0 . 2 7 4 \%$ at epoch 50 and $8 9 . 7 8 3 { \scriptstyle \pm 0 . 1 8 7 \% }$ at epoch 100, even as assigned-target accuracy rises from $9 8 . 7 6 6 \pm 0 . 0 9 8 \%$ to $9 9 . 5 0 6 \pm 0 . 2 0 6 \%$

Thus additional fitting over this interval does not produce an accompanying increase in mean validation accuracy. The endpoint true-label training–validation gap is 9.722 ± 0.175 percentage points for correct supervision. These are measured trajectory summaries, not a separately tested claim about overfitting onset.

The distinction between scale and lifetime distribution is visible in the all-layer H0 and H1-entropy profiles (Figs. 7 and 8). The birth–lifetime atlas (Fig. 9) exposes the finite intervals behind aggregate values, while geometry–topology trajectories (Fig. 13) show how persistence relates to pairwise cloud separation. Neither a visually separated curve nor a dense interval cloud establishes a statistically supported supervision effect beyond the primary analysis.

The retrospective clustering, change-rate association atlas, and temporal summary-state matrices (Figs. 14–16) provide complementary descriptions. They use complete trajectories and should not be interpreted as evidence of early predictive performance. The actual held-out evaluation remains the nested classifier: Table XVII reports its observed range and baseline-relative differences. This separation prevents visually appealing full-trajectory structure from being substituted for a leakage-controlled generalization result.

## VI. DISCUSSION AND LIMITATIONS

## A. What the Measurements Support

The benchmark separates ordinary task learning, partial fitting of a stable arbitrary assignment, and training under repeatedly reassigned targets. Their optimization outcomes and some topological summaries differ descriptively. A possible explanation is that informative targets support different directional organization than arbitrary assignments. The current measurements do not distinguish this explanation from rank, distance, scale, inactive-unit, or other geometric effects. We did not perform a matched-geometry null experiment or a causal intervention on topology.

Raw-versus-normalized disagreement is informative because it exposes dependence on representation. Normalization removes row magnitude, but can neither establish a functional interpretation nor make all layers comparable. ReLU symmetries can preserve outputs while changing coordinated weight geometry [12]. An incoming-row normalization alone does not remove effects of all compensating scalings in adjacent layers. Activation or function-based representations would be useful complementary measurements, but were not run.

The baseline result is equally important: a richly parameterized topological summary and visually distinguishable fullrun trajectories do not imply incremental prediction. PCA and clustering use all observed epochs descriptively; the early classifier uses only epochs 1–5. They answer different questions. A negative small-sample classifier result neither establishes the universal uselessness of topology nor justifies claiming that topology is predictive.

## B. Limits of Scope and Inference

Five seeds give coarse exact-test resolution and uncertain interval estimates. All configured tests fail conventional significance; this is not an equivalence finding. Unadjusted t intervals that exclude zero reflect a different assumption and multiplicity treatment from the exact/Holm analysis. Inference cannot be strengthened by counting epochs, diagram intervals, or plotted points as independent seeds.

Only one MLP, one dataset, and AdamW are studied. The arbitrary-label condition does not fully memorize within the equal 100-epoch budget. Extending only that condition would change the comparison; no such extension is claimed. The changing assignment is deliberately extreme and does not establish a general result about gradual drift, realistic relabeling, continual learning, or adaptation capacity.

Finite clouds, fixed neuron sampling, unequal layer widths, and the selected filtration conventions constrain interpretation. The H1 feature cap induces concentrated early missingness, and bottleneck analysis remains absent. There are no activation-topology, architecture, optimizer, or sampling-size replications in the empirical results. Later multidimensional visualizations are exploratory reanalyses of the same 15 models, not independent confirmation. Future work should first test representation invariance, strengthen geometry-matched baselines, increase independent seeds, and evaluate controlled adaptation tasks before expanding scientific claims.

## VII. CONCLUSION

This study characterizes persistent homology of incomingweight point clouds under correct, fixed-random, and changing-random supervision using a controlled paired design. All 15 models completed training and all 3,300 topology cases are available. Correct-label models generalize, fixedrandom models achieve partial arbitrary-target fitting, and changing-random models remain at chance under the tested setup. Normalized fc3 H1 persistence differs descriptively, but configured exact/Holm comparisons do not establish conventional significance. Early topological features do not improve condition prediction over the simpler baseline. The contribution is a reproducible empirical comparison and a transparent assessment of its limitations, leaving incremental predictive and mechanistic value unresolved.

## AVAILABILITY AND AI ASSISTANCE

Fashion-MNIST is publicly available [14]. The manuscript package includes editable sources, figure-generation code, and numerical tables derived from the archived experiment; the complete original result archive is retained separately because of its size. No public repository identifier was supplied. Generative AI assistance was used for software development, visualization, and editorial revision. Numerical findings are taken from archived measurements rather than generated experimental observations. This study uses a public clothingimage benchmark and does not involve a participant study.

## APPENDIX A

## REPRODUCIBILITY AND NUMERICAL ENVIRONMENT

The execution manifest records Linux, Python 3.13.16, a single NVIDIA Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0, torchvision 0.26.0+cu130, NumPy 2.4.6, SciPy 1.16.3, pandas 2.2.3, Ripser.py 0.6.15, Persim 0.4.1, scikit-learn 1.6.1, and Matplotlib 3.10.0. These are the archived experiment versions; later manuscript rendering uses a separate recorded environment. Training uses one CUDA device; Ripser and matching use CPU. GPU memory is PyTorch peak allocated memory, not total device occupancy.

The configuration fixes the model seeds and partition seed. Initialization uses a seeded PyTorch generator within an isolated RNG context. NumPy SeedSequence keys (s, 29, e) specify minibatch order; (s, 17, e) specify moving-label permutations, and (s, 17, 0) specifies the fixed permutation. Model loading occurs before constructing independent optimizer evolution. The setup enables deterministic PyTorch algorithms, disables cuDNN benchmarking, and sets a deterministic cuBLAS workspace configuration. Bitwise portability across different CUDA/PyTorch versions is not asserted.

The archive preserves configuration, environment, protocol, histories, target hashes, diagrams, cloud arrays, summaries, predictions, statistical results, initial states, selected epoch weights, and rolling resumable states. The compact export does not contain every per-epoch weight file, although the full runtime pipeline saves all of them. The data tensors/cache are also excluded from the compact export. Reproducing preprocessing requires regenerating the documented shared split from the official data. A separate machine-readable audit accompanies this paper.

## APPENDIX B

## DATA PARTITION AND SUPERVISION AUDIT

Stratification with partition seed 314159 yields 600 validation and 5,400 fitting examples per class from the balanced official training set. Division by 255 precedes fitting-only mean/population-standard-deviation normalization. No augmentation or validation-dependent learning-rate policy is used. The original test labels remain unchanged in every condition. Validation/test examples never receive shuffled fitting labels.

An assignment hash records the full fitting-label vector for each epoch. All five correct-label runs use their shared original vector. Each fixed-random run has one constant hash, while every changing-random run has 100 distinct hashes. Hashes support assignment reproducibility, but do not by themselves validate every possible property of the training implementation; the source code defines the separate streams and the permutation-over-examples operation.

Independent fitting targets coexist with original-label monitoring. In particular, the assigned-target accuracy of a fixedrandom model quantifies arbitrary-target fitting, whereas its original-label accuracy quantifies alignment with the actual clothing task. Near-chance true-label performance does not imply that the assigned arbitrary targets were not partly fitted. Table V retains all three accuracy roles.

## APPENDIX C

TRAINING VERIFICATION AND COMPLETE FINAL METRICS

A verification pass checks that every seed/condition has exactly epochs 1 through 100, no duplicated history epochs, finite monitored loss/norm values, and a final test record. The archived topology clouds at epoch 0 are identical across conditions within each seed/layer/representation, and the sampled indices are identical across all cases of a layer. These are direct artifact checks. Optimizer independence follows from the construction of a separate AdamW object in each run; we do not infer it merely from matching initialization clouds.

Checkpoint writes use a temporary file followed by replacement. Each epoch retains portable CPU model tensors and a rolling optimizer/model/history state; historical metadata is also recorded. A resumed epoch sequence regenerates labels and minibatches from seed and epoch. A server termination can still interrupt a write or an epoch; successfully saved Drive artifacts and a stable experiment configuration are prerequisites for recovery. The study does not claim an uninterrupted session or an independent replay of every GPU run.

## APPENDIX D

## TOPOLOGICAL DEFINITIONS AND FILTRATION CONVENTIONS

## A. Clouds and Normalization

The raw point cloud contains selected rows of a hiddenlayer matrix, with ambient dimensions 784, 1024, 512, 256, and 128 for fc1–fc5, respectively. Bias terms are not appended. The corresponding point counts are 128, 128, 128, 128, and 64. A zero incoming vector would remain zero under the 10<sup>−12</sup> denominator guard; it would not become a unit vector. The saved-cloud audit finds no such sampled row, and the minimum raw sampled row norm is approximately 0.4293.

A row permutation of a full cloud does not change its Rips diagram. A fixed-index subsample, however, must be transported under a neuron permutation to preserve the sampled object. Similarly, row normalization does not eliminate every equivalent-model scaling because compensating transformations can alter coordinates in adjacent layers. No symmetryinvariant functional measurement is claimed.

## B. Finite Intervals and Betti Curves

Each full finite-cloud Rips filtration has one essential H0 interval. It is excluded from Eq. (3), mean/maximum lifetime and matching, but remains alive in the H0 Betti curve. H1 intervals are finite when the complete finite-cloud filtration is computed. Positive finite lifetimes define feature count; there is no unreported lifetime significance threshold. Empty summaries take zero values. Entropy uses natural logarithms. Mean lifetime averages all finite intervals returned by Ripser; positive-lifetime count is reported separately.

For comparison across matched cases, each layer/representation has a shared grid. Normalized clouds have diameter at most 2, giving a fixed [0, 2] range. Raw grids use an endpoint 1.001 times the largest measured pairwise distance across all cases of that layer. Betti curves use 128 equally spaced values; landscapes use 32 values and ranks 1–3. The tent for interval $( b _ { j } , d _ { j } )$ is

TABLE V  
COMPLETE FINAL PERFORMANCE BY SEED AND SUPERVISION CONDITION.
<table><tr><td>Seed</td><td>Condition</td><td>Online loss</td><td>Assigned train (%)</td><td>True train (%)</td><td>True val. (%)</td><td>True test (%)</td></tr><tr><td>42</td><td>Correct</td><td>0.0214</td><td>99.50</td><td>99.50</td><td>89.85</td><td>89.60</td></tr><tr><td>42</td><td>Fixed random</td><td>1.0002</td><td>66.91</td><td>10.20</td><td>10.23</td><td>10.86</td></tr><tr><td>42</td><td>Changing random</td><td>2.3027</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr><tr><td>123</td><td>Correct</td><td>0.0210</td><td>99.29</td><td>99.29</td><td>89.52</td><td>89.23</td></tr><tr><td>123</td><td>Fixed random</td><td>0.8170</td><td>74.19</td><td>10.32</td><td>10.52</td><td>10.95</td></tr><tr><td>123</td><td>Changing random</td><td>2.3027</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr><tr><td>456</td><td>Correct</td><td>0.0183</td><td>99.75</td><td>99.75</td><td>89.75</td><td>89.35</td></tr><tr><td>456</td><td>Fixed random</td><td>0.9242</td><td>70.05</td><td>9.19</td><td>9.45</td><td>9.48</td></tr><tr><td>456</td><td>Changing random</td><td>2.3027</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr><tr><td>789</td><td>Correct</td><td>0.0165</td><td>99.31</td><td>99.31</td><td>89.77</td><td>89.10</td></tr><tr><td>789</td><td>Fixed random</td><td>0.9780</td><td>67.75</td><td>9.84</td><td>10.03</td><td>10.02</td></tr><tr><td>789</td><td>Changing random</td><td>2.3027</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr><tr><td>2026</td><td>Correct</td><td>0.0180</td><td>99.67</td><td>99.67</td><td>90.03</td><td>89.65</td></tr><tr><td>2026</td><td>Fixed random</td><td>1.1429</td><td>60.80</td><td>10.08</td><td>10.27</td><td>10.14</td></tr><tr><td>2026</td><td>Changing random</td><td>2.3027</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr></table>

$$
f _ { j } ( u ) = \operatorname* { m a x } \{ 0 , \operatorname* { m i n } ( u - b _ { j } , d _ { j } - u ) \} ,\tag{7}
$$

and $\lambda _ { m } ( u )$ is its mth largest value across intervals, padded with zero where necessary. The early classifier retains normalized landscapes and normalized Betti area, but excludes the raw counterparts. It does not exclude every grid-based feature.

## C. Distance Units and Computational Scope

Equation (5) specifies $p = 1$ with ground norm $q \ : = \ : 2$ Essential intervals are removed before matching. Diagram cardinality and cloud geometry affect matching sums, so neither dividing by elapsed epochs nor using normalized input rows makes all layers’ sums interchangeable. The feature cap is applied to finite diagrams after essential-interval removal. Here H0 has at most 127 finite intervals and is never capped.

For $S = 5$ seeds, $C = 3$ conditions, $K = 2 2$ checkpoints, $L \ = \ 5$ layers, $R \ = \ 2$ representations, and two homology dimensions, the evaluation count is

$$
2 L R \left[ S C ( K - 1 ) + S K { \binom { C } { 2 } } + C K { \binom { S } { 2 } } \right] = 2 6 1 0 0 .\tag{8}
$$

This accounts for 6,300 temporal, 6,600 condition, and 13,200 seed comparisons. Of 13,050 H1 comparisons, 282 are unavailable, leaving 12,768 finite H1 values; all 13,050 H0 values are finite. Bottleneck distance would characterize a maximum matching displacement under its chosen ground norm, while Wasserstein accumulates displacements. One must not claim a numerically larger L-infinity bottleneck cost than an L2- ground matching sum merely by describing a single large feature movement. No bottleneck conclusion is drawn here.

## APPENDIX E COMPLETE TOPOLOGICAL SUMMARIES AND ASSOCIATIONS

Table VI gives endpoint H0/H1 totals for every layer, condition, and representation. These final values are not substitutes for Eq. (6). Full per-case summaries remain in topology.csv; the paper package includes that machinereadable table rather than printing 3,300 rows and hundreds of landscape columns.

The archived exploratory association table contains 4,500 rows: five seeds, three conditions, five layers, two representations, three topological summaries, five monitored metrics, and two lags. It uses Spearman correlation of checkpoint rates after joining with monitored training epochs. Initialization has no training metric row and is not part of that join. Lag 0 aligns rate sequences; lag 1 shifts the topology-rate sequence once. No additional time-window restriction or higher-order detrending was applied. Undefined or insufficient correlations would be omitted rather than set to zero. Table VII reports medians over the five seed-level H1-total correlations, not a correlation obtained by pooling every epoch from every model.

## A. Training and Optimization Milestones

Tables VIII and IX report the same five paired runs at epochs 1, 5, 10, 50, and 100. Reporting both assigned-target accuracy and true-label validation accuracy separates fitting randomized supervision from learning the original task. The true-label training–validation gap is computed within seed before aggregation. In particular, high assigned-target accuracy in the fixed-random condition is not evidence of true-label generalization. The online loss averages training batches under evolving parameters, whereas the reported accuracy evaluates the end-of-epoch model; these quantities need not satisfy a deterministic one-to-one relation.

## B. Layer and Representation Sensitivity

Table X extends the primary fc3 summary to every hidden layer. Trapezoidal weighting accounts for the dense early schedule and sparse late schedule, preventing the first ten epochs from receiving disproportionate weight simply because they were sampled more frequently. These additional layer comparisons were not added to the seven-outcome primary testing family; they are descriptive sensitivity analyses.

Figures 7 and 8 distinguish connectivity-scale changes from changes in the distribution of finite loop lifetimes. H0 total persistence aggregates finite merger scales, while lifetime entropy depends on relative lifetime weights. An entropy change therefore does not necessarily imply a corresponding change in total persistence. The shaded bands summarize between-seed variability at each checkpoint and are not confidence bands for independent epochs.

TABLE VI  
LAYER-RESOLVED H0 AND H1 TOTAL PERSISTENCE AT EPOCH 100.
<table><tr><td>Layer</td><td>Condition</td><td>Raw H0</td><td>Raw H1</td><td></td><td>Normalized H0 Normalized H1</td></tr><tr><td>fc1</td><td>Correct</td><td> $3 4 3 . 0 9 9 \pm 1 2 . 2 1 0$ </td><td> $0 . 1 4 1 \pm 0 . 1 7 8$ </td><td> $1 3 1 . 8 6 7 \pm 1 . 5 1 0$ </td><td> $2 . 2 2 8 \pm 0 . 2 2 5$ </td></tr><tr><td>fc1</td><td>Fixed random</td><td> $3 3 8 . 4 3 3 \pm 2 0 . 5 0 3$ </td><td> $0 . 1 0 1 \pm 0 . 0 9 2$ </td><td> $1 2 5 . 3 4 1 \pm 3 . 8 5 6$ </td><td> $0 . 5 2 8 \pm 0 . 1 5 2$ </td></tr><tr><td>fc1</td><td>Changing random</td><td> $1 0 1 . 1 9 0 \pm 6 . 4 7 0$ </td><td> $0 . 2 9 9 \pm 0 . 0 9 3$ </td><td> $1 3 8 . 2 2 8 \pm 6 . 7 5 4$ </td><td> $0 . 3 1 6 \pm 0 . 1 8 0$ </td></tr><tr><td>fc2</td><td>Correct</td><td> $3 9 4 . 2 4 1 \pm 6 . 8 2 9$ </td><td> $0 . 0 2 7 \pm 0 . 0 6 1$ </td><td> $1 4 9 . 3 6 2 \pm 1 . 1 0 8$ </td><td> $3 . 9 2 9 \pm 0 . 2 3 3$ </td></tr><tr><td>fc2</td><td>Fixed random</td><td> $3 2 9 . 5 8 7 \pm 3 1 . 2 4 0$ </td><td> $0 . 0 1 5 \pm 0 . 0 2 0$ </td><td> $1 5 6 . 4 1 9 \pm 0 . 2 4 1$ </td><td> $1 . 7 3 8 \pm 0 . 1 8 9$ </td></tr><tr><td>fc2</td><td>Changing random</td><td> $1 0 4 . 2 9 4 \pm 4 . 1 9 6$ </td><td> $0 . 0 2 1 \pm 0 . 0 1 9$ </td><td> $1 5 8 . 5 3 9 \pm 1 . 2 3 4$ </td><td> $1 . 9 5 3 \pm 0 . 2 4 4$ </td></tr><tr><td>fc3</td><td>Correct</td><td> $3 2 1 . 4 2 4 \pm 5 . 2 3 4$ </td><td> $0 . 8 7 6 \pm 0 . 3 3 8$ </td><td> $1 3 4 . 3 3 3 \pm 0 . 8 9 8$ </td><td> $4 . 3 7 6 \pm 0 . 3 5 9$ </td></tr><tr><td>fc3</td><td>Fixed random</td><td> $3 8 1 . 6 2 4 \pm 1 8 . 6 9 8$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 7$ </td><td> $1 5 5 . 7 6 0 \pm 1 . 4 0 8$ </td><td> $2 . 7 5 5 \pm 0 . 3 9 5$ </td></tr><tr><td>fc3</td><td>Changing random</td><td> $1 0 1 . 1 0 0 \pm 3 . 4 4 2$ </td><td> $0 . 0 3 8 \pm 0 . 0 3 3$ </td><td> $1 5 3 . 8 6 2 \pm 1 . 7 2 4$ </td><td> $2 . 3 5 2 \pm 0 . 4 6 1$ </td></tr><tr><td>fc4</td><td>Correct</td><td> $1 9 6 . 4 6 6 \pm 1 . 8 0 8$ </td><td> $1 . 2 4 4 \pm 0 . 5 3 2$ </td><td> $1 2 5 . 8 7 4 \pm 1 . 6 1 9$ </td><td> $4 . 6 1 9 \pm 0 . 3 9 6$ </td></tr><tr><td>fc4</td><td>Fixed random</td><td> $3 4 2 . 5 9 8 \pm 2 6 . 1 5 1$ </td><td> $0 . 0 2 1 \pm 0 . 0 3 1$ </td><td> $1 5 0 . 3 4 3 \pm 0 . 9 3 4$ </td><td> $6 . 0 3 4 \pm 0 . 8 3 8$ </td></tr><tr><td>fc4</td><td>Changing random</td><td> $8 9 . 5 4 9 \pm 3 . 1 1 1$ </td><td> $0 . 2 7 1 \pm 0 . 2 2 3$ </td><td> $1 5 2 . 1 8 7 \pm 2 . 1 8 2$ </td><td> $3 . 8 3 6 \pm 0 . 4 7 2$ </td></tr><tr><td>fc5</td><td>Correct</td><td> $6 4 . 4 4 7 \pm 2 . 2 5 6$ </td><td> $0 . 3 5 7 \pm 0 . 3 0 5$ </td><td> $6 8 . 6 2 1 \pm 0 . 6 2 8$ </td><td> $2 . 8 8 9 \pm 0 . 2 5 7$ </td></tr><tr><td>fc5</td><td>Fixed random</td><td> $1 3 2 . 1 4 9 \pm 3 . 3 6 2$ </td><td> $0 . 0 9 8 \pm 0 . 0 3 0$ </td><td> $7 0 . 9 6 6 \pm 0 . 6 1 3$ </td><td> $3 . 3 0 3 \pm 0 . 4 0 0$ </td></tr><tr><td>fc5</td><td>Changing random</td><td> $3 8 . 3 5 0 \pm 0 . 4 4 7$ </td><td> $0 . 4 9 3 \pm 0 . 0 6 3$ </td><td> $7 4 . 5 8 5 \pm 1 . 2 7 8$ </td><td> $2 . 3 2 6 \pm 0 . 4 2 4$ </td></tr></table>

Means ± sample SD across five seeds. These are exploratory final-checkpoint summaries, not the time-averaged primary endpoint. Each representation has its own filtration units. fc1–fc4 contain 128 sampled neurons; fc5 contains 64.

TABLE VII  
EXPLORATORY NORMALIZED FC3 H1 CHANGE-RATE ASSOCIATIONS.
<table><tr><td>Condition</td><td>Lag</td><td>Loss</td><td>Gradient</td><td>Update</td><td>Val. accuracy</td><td>Weight norm</td></tr><tr><td>Correct</td><td>0</td><td>-0.096</td><td>-0.099</td><td>+0.242</td><td>-0.063</td><td>-0.042</td></tr><tr><td>Correct</td><td>1</td><td>+0.040</td><td>+0.158</td><td>+0.219</td><td>-0.212</td><td>+0.026</td></tr><tr><td>Fixed random</td><td>0</td><td>-0.358</td><td>+0.418</td><td>-0.026</td><td>+0.032</td><td>+0.138</td></tr><tr><td>Fixed random</td><td>1</td><td>-0.393</td><td>+0.505</td><td>+0.028</td><td>+0.172</td><td>+0.044</td></tr><tr><td>Changing random</td><td>0</td><td>+0.325</td><td>+0.209</td><td>+0.167</td><td>-0.055</td><td>-0.570</td></tr><tr><td>Changing random</td><td>1</td><td>+0.112</td><td>+0.391</td><td>+0.089</td><td>+0.092</td><td>-0.498</td></tr></table>

Median within-seed Spearman correlation across five seeds. Lag 1 shifts the topological change-rate sequence by one sampled checkpoint; it is not a fixed one-epoch delay. No p-values or multiplicity-adjusted causal claims are inferred.

TABLE VIII  
TRAINING MILESTONES ACROSS ALL FIVE SEEDS.
<table><tr><td>Epoch</td><td>Supervision</td><td>Online loss</td><td>Target acc. (%)</td><td>Validation (%)</td><td>True-label gap (pp)</td></tr><tr><td>1</td><td>Correct</td><td> $0 . 6 3 9 \pm 0 . 0 1 8$ </td><td> $8 4 . 0 6 4 \pm 0 . 7 7 6$ </td><td> $8 3 . 8 1 0 \pm 1 . 0 7 5$ </td><td> $0 . 2 5 4 \pm 0 . 3 4 9$ </td></tr><tr><td>1</td><td>Fixed random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 8 7 \pm 0 . 0 8 3$ </td><td> $7 . 6 8 7 \pm 3 . 7 2 2$ </td><td> $- 0 . 0 2 2 \pm 0 . 2 1 4$ </td></tr><tr><td>1</td><td>Changing random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 9 0 \pm 0 . 1 2 0$ </td><td> $1 0 . 1 3 0 \pm 1 . 9 8 1$ </td><td> $- 0 . 2 9 1 \pm 0 . 2 2 1$ </td></tr><tr><td>5</td><td>Correct</td><td> $0 . 2 8 2 \pm 0 . 0 0 3$ </td><td> $9 0 . 4 5 3 \pm 0 . 3 0 1$ </td><td> $8 8 . 5 3 7 \pm 0 . 2 9 0$ </td><td> $1 . 9 1 6 \pm 0 . 2 2 6$ </td></tr><tr><td>5</td><td>Fixed random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 1 5 1 \pm 0 . 1 8 4$ </td><td> $1 1 . 6 2 3 \pm 3 . 1 5 7$ </td><td> $- 0 . 0 6 6 \pm 0 . 2 8 0$ </td></tr><tr><td>5</td><td>Changing random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 1 7 5 \pm 0 . 0 7 3$ </td><td> $1 4 . 2 9 7 \pm 1 . 5 4 6$ </td><td> $- 0 . 0 5 2 \pm 0 . 4 4 7$ </td></tr><tr><td>10</td><td>Correct</td><td> $0 . 2 0 6 \pm 0 . 0 0 2$ </td><td> $9 2 . 9 3 1 \pm 0 . 3 2 6$ </td><td> $8 9 . 1 9 3 \pm 0 . 2 3 1$ </td><td> $3 . 7 3 8 \pm 0 . 1 9 6$ </td></tr><tr><td>10</td><td>Fixed random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 2 0 5 \pm 0 . 3 0 3$ </td><td> $9 . 4 1 3 \pm 5 . 4 8 4$ </td><td> $0 . 0 1 6 \pm 0 . 0 4 0$ </td></tr><tr><td>10</td><td>Changing random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 6 4 \pm 0 . 1 3 6$ </td><td> $1 0 . 4 6 0 \pm 2 . 8 2 1$ </td><td> $0 . 0 7 4 \pm 0 . 2 0 7$ </td></tr><tr><td>50</td><td>Correct</td><td> $0 . 0 4 0 \pm 0 . 0 0 3$ </td><td> $9 8 . 7 6 6 \pm 0 . 0 9 8$ </td><td> $8 9 . 8 3 0 \pm 0 . 2 7 4$ </td><td> $8 . 9 3 6 \pm 0 . 1 7 7$ </td></tr><tr><td>50</td><td>Fixed random</td><td> $1 . 9 0 9 \pm 0 . 1 7 0$ </td><td> $3 0 . 1 2 7 \pm 8 . 1 8 4$ </td><td> $1 0 . 6 9 0 \pm 1 . 4 2 7$ </td><td> $- 0 . 1 3 1 \pm 0 . 6 0 9$ </td></tr><tr><td>50</td><td>Changing random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>100</td><td>Correct</td><td> $0 . 0 1 9 \pm 0 . 0 0 2$ </td><td> $9 9 . 5 0 6 \pm 0 . 2 0 6$ </td><td> $8 9 . 7 8 3 \pm 0 . 1 8 7$ </td><td> $9 . 7 2 2 \pm 0 . 1 7 5$ </td></tr><tr><td>100</td><td>Fixed random</td><td> $0 . 9 7 2 \pm 0 . 1 1 9$ </td><td> $6 7 . 9 3 8 \pm 4 . 8 8 7$ </td><td> $1 0 . 1 0 0 \pm 0 . 4 0 2$ </td><td> $- 0 . 1 7 4 \pm 0 . 0 8 5$ </td></tr><tr><td>100</td><td>Changing random</td><td> $2 . 3 0 3 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $1 0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

Entries are mean ± sample SD across five seeds. The gap is true-label training accuracy minus validation accuracy within each seed; it differs from assigned-target accuracy under randomized labels.

but does not fix pairwise angular separation or the cloud’s singular spectrum. Table XII quantifies within-seed agreement of raw and normalized H1 trajectories. A large positive correlation can coexist with different absolute persistence levels; correlation is not an invariance guarantee and can arise from a common training trend.

Table XI places endpoint topology beside sampled-cloud geometry. Row normalization controls individual row length

Figure 9 shows how aggregate summaries relate to the actual intervals. Long-lived intervals appear farther above the horizontal axis, while horizontal displacement changes birth scale. A denser panel may contain more short intervals rather than stronger persistent structure. This view supplements lifetime totals without imposing a new significance threshold or deleting low-persistence features.

TABLE IX  
OPTIMIZATION MILESTONES AND RECORDED EPOCH DURATION.
<table><tr><td>Epoch</td><td>Supervision</td><td>Gradient norm</td><td>Weight norm</td><td>Update norm</td><td>Epoch duration (s)</td></tr><tr><td>1</td><td>Correct</td><td> $4 . 2 9 9 \pm 0 . 6 9 7$ </td><td> $3 2 . 2 8 5 \pm 0 . 1 2 0$ </td><td> $1 8 . 5 3 3 \pm 0 . 2 1 7$ </td><td> $1 . 7 4 7 \pm 0 . 3 3 8$ </td></tr><tr><td>1</td><td>Fixed random</td><td> $0 . 0 8 0 \pm 0 . 0 0 3$ </td><td> $2 7 . 1 7 7 \pm 0 . 0 6 7$ </td><td> $9 . 2 1 1 \pm 0 . 1 6 7$ </td><td> $1 . 8 2 6 \pm 0 . 3 4 4$ </td></tr><tr><td>1</td><td>Changing random</td><td> $0 . 0 8 1 \pm 0 . 0 0 3$ </td><td> $2 7 . 2 2 3 \pm 0 . 1 0 6$ </td><td> $9 . 3 6 0 \pm 0 . 3 2 4$ </td><td> $1 . 8 6 4 \pm 0 . 3 8 5$ </td></tr><tr><td>5</td><td>Correct</td><td> $1 . 0 0 1 \pm 0 . 0 4 7$ </td><td> $4 6 . 0 0 0 \pm 0 . 2 2 2$ </td><td> $9 . 2 4 9 \pm 0 . 2 3 5$ </td><td> $1 . 8 6 2 \pm 0 . 3 6 2$ </td></tr><tr><td>5</td><td>Fixed random</td><td> $0 . 0 6 7 \pm 0 . 0 0 1$ </td><td> $3 0 . 6 0 3 \pm 1 . 0 1 0$ </td><td> $7 . 6 7 3 \pm 1 . 3 6 8$ </td><td> $1 . 5 9 8 \pm 0 . 0 6 0$ </td></tr><tr><td>5</td><td>Changing random</td><td> $0 . 0 6 6 \pm 0 . 0 0 1$ </td><td> $2 9 . 5 6 3 \pm 0 . 2 5 4$ </td><td> $6 . 1 7 7 \pm 0 . 4 0 4$ </td><td> $1 . 6 5 9 \pm 0 . 1 9 3$ </td></tr><tr><td>10</td><td>Correct</td><td> $0 . 8 2 7 \pm 0 . 0 4 3$ </td><td> $5 7 . 6 6 3 \pm 0 . 3 4 0$ </td><td> $9 . 4 9 1 \pm 0 . 1 1 4$ </td><td> $1 . 8 5 1 \pm 0 . 4 6 8$ </td></tr><tr><td>10</td><td>Fixed random</td><td> $0 . 0 6 7 \pm 0 . 0 0 3$ </td><td> $3 6 . 0 5 9 \pm 2 . 3 9 1$ </td><td> $7 . 9 7 8 \pm 1 . 5 3 6$ </td><td> $2 . 0 3 4 \pm 0 . 3 6 4$ </td></tr><tr><td>10</td><td>Changing random</td><td> $0 . 0 6 4 \pm 0 . 0 0 2$ </td><td> $3 2 . 4 2 0 \pm 1 . 0 5 1$ </td><td> $6 . 0 1 7 \pm 0 . 6 9 9$ </td><td> $1 . 8 4 4 \pm 0 . 4 2 2$ </td></tr><tr><td>50</td><td>Correct</td><td> $0 . 5 1 8 \pm 0 . 0 1 3$ </td><td> $1 0 2 . 7 5 1 \pm 0 . 6 3 6$ </td><td> $1 0 . 3 5 3 \pm 1 . 3 9 1$ </td><td> $1 . 5 8 5 \pm 0 . 0 9 8$ </td></tr><tr><td>50</td><td>Fixed random</td><td> $1 . 1 7 2 \pm 0 . 3 6 1$ </td><td> $1 0 8 . 2 0 3 \pm 5 . 4 7 1$ </td><td> $1 0 . 6 4 4 \pm 1 . 4 8 8$ </td><td> $1 . 9 1 1 \pm 0 . 5 1 0$ </td></tr><tr><td>50</td><td>Changing random</td><td> $0 . 0 5 8 \pm 0 . 0 0 1$ </td><td> $3 6 . 5 7 4 \pm 2 . 4 3 6$ </td><td> $0 . 0 8 9 \pm 0 . 0 2 5$ </td><td> $1 . 5 6 7 \pm 0 . 0 8 0$ </td></tr><tr><td>100</td><td>Correct</td><td> $0 . 3 5 8 \pm 0 . 0 2 1$ </td><td> $1 2 8 . 6 9 2 \pm 0 . 8 5 5$ </td><td> $1 0 . 3 8 9 \pm 2 . 2 1 3$ </td><td> $1 . 8 6 1 \pm 0 . 4 7 7$ </td></tr><tr><td>100</td><td>Fixed random</td><td> $3 . 7 4 5 \pm 0 . 3 5 2$ </td><td> $1 3 5 . 9 3 2 \pm 3 . 5 4 3$ </td><td> $7 . 9 8 8 \pm 0 . 3 5 1$ </td><td> $1 . 7 6 2 \pm 0 . 3 3 8$ </td></tr><tr><td>100</td><td>Changing random</td><td> $0 . 0 5 7 \pm 0 . 0 0 1$ </td><td> $3 4 . 2 3 6 \pm 3 . 0 0 8$ </td><td> $0 . 0 8 3 \pm 0 . 0 2 3$ </td><td> $1 . 7 3 8 \pm 0 . 2 1 4$ </td></tr></table>

Gradient norm is the monitored pre-clipping norm. Weight and update norms are global Euclidean norms. Recorded epoch duration excludes checkpoint I/O; no full-pipeline runtime is inferred.

TABLE X  
TIME-WEIGHTED H1 TOTAL PERSISTENCE BY LAYER AND REPRESENTATION.
<table><tr><td>Layer</td><td>Supervision</td><td>Raw</td><td>Normalized</td></tr><tr><td>fc1</td><td>Correct</td><td> $0 . 2 8 1 \pm 0 . 0 8 7$ </td><td> $3 . 1 3 3 \pm 0 . 4 2 9$ </td></tr><tr><td>fc1</td><td>Fixed random</td><td> $0 . 2 0 6 \pm 0 . 0 6 4$ </td><td> $0 . 8 4 8 \pm 0 . 1 2 6$ </td></tr><tr><td>fc1</td><td>Changing random</td><td> $0 . 3 9 3 \pm 0 . 0 7 2$ </td><td> $0 . 3 6 4 \pm 0 . 1 0 2$ </td></tr><tr><td>fc2</td><td>Correct</td><td> $0 . 3 0 2 \pm 0 . 0 8 6$ </td><td> $3 . 9 7 9 \pm 0 . 2 4 1$ </td></tr><tr><td>fc2</td><td>Fixed random</td><td> $0 . 0 6 9 \pm 0 . 0 3 0$ </td><td> $1 . 6 0 8 \pm 0 . 1 4 6$ </td></tr><tr><td>fc2</td><td>Changing random</td><td> $0 . 0 5 8 \pm 0 . 0 1 8$ </td><td> $1 . 9 7 3 \pm 0 . 1 7 2$ </td></tr><tr><td>fc3</td><td>Correct</td><td> $1 . 3 1 4 \pm 0 . 1 7 5$ </td><td> $4 . 3 5 9 \pm 0 . 2 0 4$ </td></tr><tr><td>fc3</td><td>Fixed random</td><td> $0 . 1 1 0 \pm 0 . 0 3 1$ </td><td> $2 . 7 4 3 \pm 0 . 2 2 7$ </td></tr><tr><td>fc3</td><td>Changing random</td><td> $0 . 1 1 0 \pm 0 . 0 2 8$ </td><td> $2 . 5 7 5 \pm 0 . 4 0 8$ </td></tr><tr><td>fc4</td><td>Correct</td><td> $1 . 5 5 4 \pm 0 . 3 1 8$ </td><td> $5 . 2 1 5 \pm 0 . 1 4 5$ </td></tr><tr><td>fc4</td><td>Fixed random</td><td> $0 . 3 0 6 \pm 0 . 0 6 5$ </td><td> $5 . 1 3 4 \pm 0 . 5 4 0$ </td></tr><tr><td>fc4</td><td>Changing random</td><td> $0 . 4 9 2 \pm 0 . 2 1 7$ </td><td> $4 . 1 1 5 \pm 0 . 3 7 8$ </td></tr><tr><td>fc5</td><td>Correct</td><td> $0 . 6 8 6 \pm 0 . 0 8 4$ </td><td> $3 . 1 3 0 \pm 0 . 2 3 5$ </td></tr><tr><td>fc5</td><td>Fixed random</td><td> $0 . 3 1 5 \pm 0 . 0 5 2$ </td><td> $2 . 7 2 4 \pm 0 . 2 1 9$ </td></tr><tr><td>fc5</td><td>Changing random</td><td> $0 . 6 4 7 \pm 0 . 0 5 5$ </td><td> $2 . 4 2 2 \pm 0 . 3 9 6$ </td></tr></table>

Within each seed, trapezoidal integration over the actual 22 checkpoint epochs is divided by 100. Entries summarize these five seed-level values as mean ± sample SD. Cross-layer scales are not interchangeable.

TABLE XI  
FINAL FC3 GEOMETRY AND FINITE H1 STRUCTURE.
<table><tr><td>Supervision</td><td>Cloud</td><td>Mean distance</td><td>Effective rank</td><td>Leading singular value</td><td>H1 count</td><td>Max. lifetime</td></tr><tr><td>Correct</td><td>raw</td><td> $3 . 6 2 9 \pm 0 . 0 3 3$ </td><td> $1 0 7 . 5 5 6 \pm 0 . 5 8 0$ </td><td> $9 . 1 3 4 \pm 0 . 2 4 3$ </td><td> $1 2 . 0 0 0 \pm 3 . 1 6 2$ </td><td> $0 . 2 5 5 \pm 0 . 0 7 8$ </td></tr><tr><td>Correct</td><td>normalized</td><td> $1 . 3 5 2 \pm 0 . 0 0 3$ </td><td> $1 1 0 . 7 9 6 \pm 0 . 4 0 0$ </td><td> $3 . 5 7 7 \pm 0 . 0 7 9$ </td><td> $1 2 7 . 6 0 0 \pm 8 . 1 7 3$ </td><td> $0 . 1 6 9 \pm 0 . 0 1 8$ </td></tr><tr><td>Fixed random</td><td>raw</td><td> $4 . 2 8 7 \pm 0 . 1 9 1$ </td><td> $1 1 3 . 1 5 1 \pm 3 . 0 5 8$ </td><td> $8 . 3 9 3 \pm 0 . 4 3 8$ </td><td> $0 . 4 0 0 \pm 0 . 5 4 8$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 7$ </td></tr><tr><td>Fixed random</td><td>normalized</td><td> $1 . 3 7 2 \pm 0 . 0 0 4$ </td><td> $1 1 9 . 1 4 8 \pm 0 . 5 2 8$ </td><td> $3 . 2 1 6 \pm 0 . 2 0 5$ </td><td> $1 6 9 . 6 0 0 \pm 1 9 . 2 8 2$ </td><td> $0 . 0 6 9 \pm 0 . 0 1 0$ </td></tr><tr><td>Changing random</td><td>raw</td><td> $0 . 9 3 9 \pm 0 . 0 4 7$ </td><td> $1 1 7 . 8 4 4 \pm 0 . 6 7 7$ </td><td> $3 . 0 0 8 \pm 0 . 3 1 6$ </td><td> $5 . 4 0 0 \pm 4 . 7 2 2$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 8$ </td></tr><tr><td>Changing random</td><td>normalized</td><td> $1 . 3 1 9 \pm 0 . 0 1 5$ </td><td> $1 1 9 . 0 1 8 \pm 0 . 5 3 2$ </td><td> $4 . 3 1 4 \pm 0 . 2 7 3$ </td><td> $1 9 4 . 2 0 0 \pm 1 9 . 6 9 0$ </td><td> $0 . 0 5 0 \pm 0 . 0 0 9$ </td></tr></table>

All entries are mean ± sample SD across seeds. Spectral quantities use the sampled cloud, not the full weight matrix. Effective rank uses normalized singular values, not squared singular values.

## APPENDIX F

## DISTANCE COVERAGE AND MISSINGNESS

Tables XIII and XIV decompose every skipped matching calculation. All are normalized H1 comparisons in fc1–fc4, and all involve the first recorded epochs. Capping is therefore not missing completely at random with respect to diagram size or training time. A temporal endpoint averaged over available rates may differ from an endpoint defined on all intended intervals. This limitation is retained even though every point cloud and persistence diagram is present.

TABLE XII  
WITHIN-SEED RAW VERSUS NORMALIZED H1 TRAJECTORY AGREEMENT.
<table><tr><td>Layer</td><td>Supervision</td><td>Median Spearman</td><td>Minimum</td><td>Maximum</td></tr><tr><td>fc1</td><td>Correct</td><td>0.630</td><td>0.421</td><td>0.759</td></tr><tr><td>fc1</td><td>Fixed random</td><td>0.142</td><td>-0.101</td><td>0.675</td></tr><tr><td>fc1</td><td>Changing random</td><td>0.570</td><td>0.526</td><td>0.936</td></tr><tr><td>fc2</td><td>Correct</td><td>0.173</td><td>-0.185</td><td>0.311</td></tr><tr><td>fc2</td><td>Fixed random</td><td>0.508</td><td>0.436</td><td>0.599</td></tr><tr><td>fc2</td><td>Changing random</td><td>0.632</td><td>-0.123</td><td>0.850</td></tr><tr><td>fc3</td><td>Correct</td><td>0.469</td><td>-0.136</td><td>0.630</td></tr><tr><td>fc3</td><td>Fixed random</td><td>0.730</td><td>0.617</td><td>0.829</td></tr><tr><td>fc3</td><td>Changing random</td><td>0.942</td><td>0.819</td><td>0.951</td></tr><tr><td>fc4</td><td>Correct</td><td>0.623</td><td>0.225</td><td>0.845</td></tr><tr><td>fc4</td><td>Fixed random</td><td>0.512</td><td>0.328</td><td>0.877</td></tr><tr><td>fc4</td><td>Changing random</td><td>0.907</td><td>0.851</td><td>0.953</td></tr><tr><td>fc5</td><td>Correct</td><td>0.571</td><td>0.362</td><td>0.851</td></tr><tr><td>fc5</td><td>Fixed random</td><td>0.448</td><td>0.149</td><td>0.855</td></tr><tr><td>fc5</td><td>Changing random</td><td>0.814</td><td>0.232</td><td>0.916</td></tr></table>

Correlations use 22 matched checkpoints per seed. The range is across five seeds. These descriptive associations may reflect shared training trends; they are not independent-observation significance tests.

TABLE XIII  
UNAVAILABLE H1 DISTANCES BY LAYER AND COMPARISON TYPE.
<table><tr><td>Layer</td><td>Temporal</td><td>Conditions</td><td>Seeds</td><td>Total</td></tr><tr><td>fc1</td><td>15</td><td>15</td><td>30</td><td>60</td></tr><tr><td>fc2</td><td>15</td><td>15</td><td>30</td><td>60</td></tr><tr><td>fc3</td><td>16</td><td>17</td><td>34</td><td>67</td></tr><tr><td>fc4</td><td>21</td><td>26</td><td>48</td><td>95</td></tr><tr><td>Total</td><td>67</td><td>73</td><td>142</td><td>282</td></tr></table>

All 282 skips are in normalized clouds; raw H1 and all H0 distances are available. No fc5 comparison was skipped. A skip occurs when either finite diagram contains more than 400 intervals.

TABLE XIV  
SKIPPED H1 COMPARISONS BY RECORDED EPOCH.
<table><tr><td>Epoch</td><td>Temporal</td><td>Conditions</td><td>Seeds</td><td>Total</td></tr><tr><td>0</td><td>0</td><td>60</td><td>120</td><td>180</td></tr><tr><td>1</td><td>60</td><td>13</td><td>22</td><td>95</td></tr><tr><td>2</td><td>7</td><td>0</td><td>0</td><td>7</td></tr></table>

Temporal rows are assigned to the later endpoint; condition and seed comparisons use their common endpoint. The decomposition does not count a missing distance as zero.

Bottleneck values in the distance table are missing by design. A computationally deferred metric is not a measured zero. The numerical summaries, figures, and prediction inputs use the available Wasserstein values and identify caps in the status column. During classification, median imputation is learned within each training fold; it does not reclassify an unavailable raw distance as an experimentally observed zero.

## APPENDIX G

## COMPLETE STATISTICAL FAMILY AND PER-SEED EFFECTS

The original configured family has seven tests, all listed in Table III. Three pairwise condition contrasts use timeaveraged normalized fc3 H1 total persistence, three use the corresponding mean available temporal Wasserstein rate, and one compares topology and baseline classification. Table XV gives the complete five-seed effect vectors, with orientations changed only for readability; reversing a contrast flips its mean and interval but not its two-sided $p$ value.

For a test statistic $T = | \bar { \delta } |$ , the exact sign-flip value is

$$
p = \frac { 1 } { 3 2 } \sum _ { \epsilon \in \{ - 1 , 1 \} ^ { 5 } } { \bf 1 } \left\{ \left| \frac { 1 } { 5 } \sum _ { s = 1 } ^ { 5 } \epsilon _ { s } \delta _ { s } \right| \geq | \bar { \delta } | \right\} .\tag{9}
$$

The implementation uses a $1 0 ^ { - 1 4 }$ numerical tolerance in the comparison. Sign symmetry is an assumption of the null distribution, not something guaranteed merely by a paired design. Two-sided resolution is coarse. All five differences have the same sign in each correct-versus-random persistence contrast, producing $p = 0 . 0 6 2 5$ . The two smallest values in the seven-test family become 0.4375 after Holm monotonicity adjustment. Raw and adjusted values are computed from actual differences, not inferred from the adjusted result alone.

The intervals in Table III and Fig. 5 are $t _ { 4 }$ intervals, not percentile-bootstrap or permutation confidence intervals. They are unadjusted and rely on a different distributional approximation. Figure 5 presents these intervals together with the individual seed effects to expose uncertainty rather than substituting parametric intervals for the configured exact tests. Outer-fold classification effects share fitted training observations across folds, so their reported interval and sign-flip value are descriptive.

## APPENDIX H

## PREDICTION FEATURES, SPLITS, AND REGULARIZATION

The nominal topology vector has 1,090 coordinates. Each of five normalized layers supplies ten finite-lifetime summaries (five each for H0 and H1), two Betti areas, and $2 \times 3 \times 3 2 =$ 192 landscape coordinates, giving $5 \times 2 0 4 = 1 0 2 0$ . Five raw layers contribute only the ten finite-lifetime summaries, giving 50. Temporal Wasserstein-rate summaries for five layers, two representations, and two homology dimensions contribute 20.

![](images/f2f3845a494ae27fc8f34d5f4aeb5dac70b1cc8401ddec7cadc59d37fcb307b9.jpg)

![](images/fa68a4abbb6f049ca094c93d8e1d4f66d76c7b7c63ee17cad29a5ab90ab94e0a.jpg)

![](images/1a8bad528bd7be856a99501795413188cc98e9a7f0310d2ee147447c5dc30c14.jpg)

![](images/9e21e51ea10130a131015d40e8c06c9bd08fc091f808f34652d95266012f1e11.jpg)

![](images/8bdf3e0c30ef80b2408823e0109fafbfd601c62018bfdd681a9ba04752b8f953.jpg)

![](images/5afb5f68bf6fbee9fc3981935333573f51f5d0da589279d203ebc51b8cc63092.jpg)

![](images/dab8efe317ed2b0cef36b6c439c9526001d0944cd1bf3217c2bcbf0887820ba5.jpg)

![](images/10afe29d5d4f960d6e888adb72b02cb8bcc2f5fb2ac698308d651f621a08a847.jpg)

![](images/7117f90dd8e9549de2b7214d15679b0f50258b7f0731849d5b8dcba391a09ca9.jpg)

![](images/62ff3495404a14c105be0565a543ad7f9a2c99c798f99c8ae88d230320683ed8.jpg)  
Fig. 7. H0 total persistence for all five hidden layers, in raw and normalized incoming-weight clouds. Lines are five-seed means; shading is one sample standard deviation. Colors follow the main-text supervision convention. Each panel retains its own vertical scale to expose within-layer evolution.

Each coordinate is averaged over the eligible early checkpoints rather than retaining five separate time positions.

The 94 baseline coordinates are nine geometric quantities for each of ten layer/representation combinations (90 coordinates) plus four training averages. Geometry comprises rownorm mean/SD; pairwise-distance mean/SD/maximum; three leading singular values; and effective rank of the sampled cloud matrix. If its singular values are $\sigma _ { i } ,$ effective rank is

$$
\exp \left( - \sum _ { i } q _ { i } \log q _ { i } \right) , \qquad q _ { i } = \sigma _ { i } / \sum _ { j } \sigma _ { j } .\tag{10}
$$

This uses singular values, not squared energy fractions. The four training quantities are loss, pre-clipping gradient norm, global weight norm, and update norm. Neither validation/test accuracy nor class identity is a predictor. Combined dimensionality is 1,184. Constant or sparse coordinates are retained by the specified imputer/scaler rather than used to claim a lower-dimensional effective feature budget.

![](images/100ab7fd7b9310acd438e747962e32bd40625b3458ac353c7a10d157af3b3ef0.jpg)

![](images/f8809e0e42ae3bea56d7f29bbf4059aeaa71cd66739d922c3023b5ec399cbe02.jpg)

![](images/7e3ad048148093c00d9724db173a90593d2bfeb45488f1a10b7935011780d06d.jpg)

![](images/4015b6a6044f0ce2449f68a78876732fc4adfe8fa8deaf94689c597b58487df0.jpg)

![](images/ed7671178b74bb18795d9df2a61ec0c906d08eeb933bbfa1b1254763f26de0cd.jpg)

![](images/44b9889c8c18bf85a277ff2c03e75594dec0dacaf17539eae6bfaf498de7c042.jpg)

![](images/29be2f0edf49fce8856bb0efa3a8f9e48ff6ae53dfddcbe276ea6cd301231741.jpg)

![](images/7ac500f251e27081e40772117ea239805216b1eff28e2bbed6e592547c4fb222.jpg)

![](images/c4072502274e838971094a118adf212aeb183d8433f94a2e4442c56d5814e4a0.jpg)

![](images/a08da6db623ec84df6377a36c14c9a01f8b9f088faf52484e2ce6e2cc05f65b3.jpg)  
Fig. 8. Finite H1 lifetime entropy throughout training. Each row is one hidden layer; columns distinguish raw and normalized clouds. Five-seed mean and sample standard deviation are shown. Entropy is dimensionless and summarizes normalized lifetime weights, rather than the magnitude of persistence.

Each outer fold holds out one seed’s three examples, leaving 12 training examples from four seeds. Inner leave-one-seed-out selection therefore fits on nine examples and validates on three at each split. The pipeline uses training-fold median imputation with empty-feature retention, StandardScaler, and multinomial logistic regression with L2 regularization, the L-BFGS solver, and 3,000 maximum iterations. The $C$ grid is {0.01, 0.1, 1, 10} and selection maximizes mean inner balanced accuracy. Table XVI gives all selected values.

![](images/a9efa3870bf1a577644ae178e87c939118c01915a83ffd0a3513742596234db7.jpg)  
Fig. 9. Birth–lifetime views of all finite H1 intervals for normalized fc3 at epochs 0, 10, and 100. Each panel pools five seeds solely for visualization; points are not treated as independent experimental replicates. Axes are shared across the nine panels. Overplotting and diagram cardinality influence apparent density Initialization panels coincide because paired clouds are identical within seed.

TABLE XV  
ALL PER-SEED EFFECTS IN THE CONFIGURED COMPARISON FAMILY.
<table><tr><td>Outcome</td><td>Comparison</td><td>42</td><td>123</td><td>456</td><td>789</td><td>2026</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Fixed random – Changing random</td><td>-0.0781</td><td>0.2515</td><td>0.2400</td><td>0.7685</td><td>-0.3417</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Correct — Fixed random</td><td>1.3069</td><td>1.9862</td><td>1.5417</td><td>1.6765</td><td>1.5666</td></tr><tr><td> $\overline { { \mathrm { T P } } } _ { 1 }$ </td><td>Correct — Changing random</td><td>1.2287</td><td>2.2378</td><td>1.7817</td><td>2.4449</td><td>1.2249</td></tr><tr><td> $\overline { { r } } _ { 1 }$ </td><td>Fixed random – Changing random</td><td>0.2444</td><td>0.0586</td><td>-0.0648</td><td>0.1408</td><td>0.0973</td></tr><tr><td> ${ \overline { { r } } } _ { 1 }$ </td><td>Correct — Fixed random</td><td>-0.0473</td><td>-0.0007</td><td>0.0246</td><td>-0.1077</td><td>-0.0722</td></tr><tr><td> ${ \overline { { r } } } _ { 1 }$ </td><td>Correct — Changing random</td><td>0.1970</td><td>0.0579</td><td>-0.0402</td><td>0.0331</td><td>0.0251</td></tr><tr><td> $\Delta \mathrm { B A }$ </td><td>Topology – Baseline</td><td>33.3333</td><td>-33.3333</td><td>0.0000</td><td>-33.3333</td><td>0.0000</td></tr></table>

Orientations match Table III; classification differences are percentage points. No epoch is used as an independent replicate.

Topology has nine correct held-out condition assignments in aggregate, and baseline has ten, out of 15; combined also has nine. These totals follow from one example per class in each fold. The archive does not contain the individual predicted class labels, so no model-specific confusion matrix, classwise precision, or per-example explanation is fabricated. The saved fold table is the complete available predictive result. Detailed feature names and the reconstructed feature matrix are included in the source package; reconstruction does not refit classifiers or modify the recorded scores.

The descriptive range in Table XVII makes the coarseness of three-example test folds explicit: a single changed prediction moves a fold’s balanced accuracy by 33.33 percentage points.

TABLE XVI  
INNER-VALIDATION CHOICES OF INVERSE REGULARIZATION STRENGTH C.
<table><tr><td>Held-out seed</td><td>Topology</td><td>Baseline</td><td>Combined</td></tr><tr><td>42</td><td>0.1</td><td>1</td><td>0.01</td></tr><tr><td>123</td><td>0.01</td><td>0.1</td><td>0.01</td></tr><tr><td>456</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>789</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>2026</td><td>0.01</td><td>0.01</td><td>0.01</td></tr></table>

All searches used C ∈ {0.01, 0.1, 1, 10}. Smaller C means stronger L2 regularization; selection used only inner training folds.

The topology and combined feature families each average 60%, versus 66.67% for the geometric/optimization baseline. Neither the larger coordinate budget nor combining feature families produced an observed gain in this small evaluation. This does not establish that topology is uninformative in larger data regimes.

## APPENDIX I

## EXTENDED VISUALIZATIONS AND THEIR INTERPRETATION

Figure 10 summarizes normalized fc3 trajectories using six features: H0 total persistence and entropy; H1 total persistence, entropy, maximum lifetime, and feature count. A log(1 + x) transform is applied to the nonnegative magnitude/count coordinates, and all six features are standardized jointly over all saved seed/condition/checkpoint observations. Full-data PCA is a descriptive visualization, not part of held-out early prediction. Thin paths show seeds, thick paths show condition means, and endpoint markers identify measured epochs.

Figure 11 links topological, geometric, and optimization change-rate diagnostics. Each edge uses a median within-seed Spearman coefficient, an absolute median threshold of 0.25, and at least four same-sign seeds. Width represents coefficient magnitude; blue and red indicate positive and negative association. These are exploratory plotting thresholds, not significance thresholds or directed effects. Leading singular value is used instead of a normalized row-norm diagnostic that is nearly constant by construction.

Figure 12 provides rank-one persistence landscape surfaces on a shared display grid calculated from the saved finite H1 diagrams. Averaging is performed over seeds after computing each diagram’s tent envelope. Surface polygons connect measured epochs for display and do not create new observations. Both axis units and height scale are shared across conditions.

The accompanying exploratory figure collection additionally contains temporal feature-distance matrices, layer-time contrasts, hierarchy-based signature plots, multivariate scatter matrices, three-dimensional dispersion ellipsoids, and circular atlases. Their conventions are documented in their captions. Standardized heatmaps do not make layer filtrations physically equivalent. Ellipsoids of five seed points describe covariance geometry, not a 95% confidence region. Average-linkage clustering and full-data PCA do not supply independent replications. Visualization complexity does not add experimental evidence.

## A. Geometry–Topology Trajectories

Figure 13 tests a visual interpretation of the topology curves against an elementary geometric scale. Normalization removes row magnitude, but a normalized trajectory may still move in mean pairwise distance. A bend or loop in this projection does not establish hysteresis, an attractor, or chaos: the visualization omits other state variables and averages distinct seed trajectories.

## B. Multivariate Seed-Level Signatures

The clustering in Fig. 14 summarizes complete trajectories, including late training. It is not an early-prediction evaluation and its full-data standardization must not be reused as a preprocessing step in the held-out classifier. Featurewise standardization emphasizes relative run variation and removes absolute cross-feature units; the dendrogram is not an uncertainty estimate or a phylogenetic relationship.

## C. Layer-Wise Association Structure

Figure 15 expands the association results across depth and topological summaries. The input is the archived change-rate association table, not a new pooled correlation over all model checkpoints. Sign agreement is a descriptive reproducibility cue. Temporal dependence, numerous displayed cells, and the shared influence of training time prevent interpreting isolated large correlations as corrected discoveries or causal links.

## D. Temporal Separation of Summary States

The symmetric matrices in Fig. 16 compare early and late states without assuming linear trajectories. Their zero diagonal is mathematical self-distance. Block structure can indicate similar summary states over a range of checkpoints, but does not establish mechanistic convergence or equal persistence diagrams: six summaries are a lossy representation. Global standardization is used only for retrospective visualization. No additional hypothesis test or classifier score is inferred from these matrices.

## APPENDIX J EXECUTION COST, ARTIFACT ACCESS, AND REPRODUCTION

Summing recorded epoch runtimes gives 2,643.82 s (44.06 min) for training and end-of-epoch monitoring across the 15 runs. These timers end before persistent checkpoint writes and are not end-to-end Colab wall-clock time. Summing cached topology timers gives 191.48 s (3.19 min); those timers cover distance construction, Ripser, and singularvalue/summary calculation, not all cloud I/O, grid encoding, matching, plots, or report export. Separate wall-clock timers for diagram matching and nested classification were not recorded and cannot be reconstructed from result values. The supplied compressed archive is 2,192,571,623 bytes (2.19 GB decimal; approximately 2.04 GiB).

The source package contains the user draft for traceability, official IEEEtran files, LaTeX sections/tables, 600-DPI PNG figure assets, plotting scripts, numerical CSVs, and audit records. It does not duplicate the approximately 2.19 GB original ZIP or the model checkpoints. The companion reproduction code installs the archived modular source and explicitly retains the Wasserstein-only/deferred-bottleneck execution choice. Example commands are:

TABLE XVII  
DESCRIPTIVE SUMMARY OF THE FIVE HELD-OUT-SEED FOLDS.
<table><tr><td>Feature family</td><td>Coordinates</td><td>Balanced acc. (%)</td><td>Minimum</td><td>Maximum</td><td>Versus baseline (pp)</td></tr><tr><td>baseline</td><td>94</td><td> $6 6 . 6 6 7 \pm 2 3 . 5 7 0$ </td><td>33.33</td><td>100.00</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>combined</td><td>1184</td><td> $6 0 . 0 0 0 \pm 1 4 . 9 0 7$ </td><td>33.33</td><td>66.67</td><td> $- 6 . 6 6 7 \pm 2 7 . 8 8 9$ </td></tr><tr><td>topology</td><td>1090</td><td> $6 0 . 0 0 0 \pm 1 4 . 9 0 7$ </td><td>33.33</td><td>66.67</td><td> $- 6 . 6 6 7 \pm 2 7 . 8 8 9$ </td></tr></table>

Mean ± sample SD describes the five folds; overlapping training sets prevent interpreting folds as independent replications. Each test fold contains three models. Per-class predictions were not saved, so no confusion matrix is reconstructed.

![](images/966f7eb19b493a7771c3e156156abbf0cce97b006503f55d246189107dfe2b3b.jpg)

![](images/715ffe11606b40899887e7e299ad3696639b3e14d2bcaf0f6e1f66953ed80433.jpg)

![](images/eda96b5813b53cca17dd9a5f14e5b99fff5cffad2174941e1770d48f42d9bea0.jpg)

Fig. 10. Exploratory normalized fc3 topological-feature trajectories in a common PCA space. Six standardized summaries are used, with log transforms on magnitude/count coordinates. PC1 and PC2 explain 73.5% and 22.3% of the full descriptive variance, respectively. Thin lines show seeds, thick lines show condition means; circles, triangles, and squares identify epochs 0, 10, and 100. Apparent separation does not establish a mechanism, a phase transition, or held-out predictive gain.  
![](images/28d9ae7e101b80996a8177b3ed03402ad9b31bc4776fcb531cf06b32627e133c.jpg)  
Fig. 11. Exploratory association ribbons for normalized fc3. H0T/H1T: total persistence; H1E: entropy; Rank: effective rank; Dist.: mean pairwise distance; S1: leading singular value; Loss: online loss; Grad.: pre-clipping gradient norm; Upd.: epoch update norm; Val.: validation accuracy. Ribbons require |median ρ| ≥ 0.25 and sign agreement in at least four seeds. They represent undirected associations, not neuron connectivity or causal circuits.

```batch
python -m pip install -r requirements.txt
python reproduce.py -mode pilot
python reproduce.py -mode full
python reproduce.py -mode full
-analysis-only
```

These commands apply to the bundled reproduction project and a full runtime artifact directory. A compact selectedweight export is not an all-epoch training backup; rebuilding every omitted checkpoint from it would require restored full weights or retraining. The manuscript’s numerical tables and figures can instead be regenerated from the supplied CSVs, with the original ZIP required only for audits or displays that read cloud/diagram arrays. No additional training was performed in preparing this paper.

![](images/8820f7d069189db27c2e135a4ce8387f476652b7fa020208eefa5d4b864ff8da.jpg)

Fig. 12. Exploratory rank-one H1 persistence landscapes in normalized fc3 clouds. Heights average the first landscape separately computed for each seed A common display grid and height scale are used; epoch positions use actual training time. The visualization includes all finite diagrams, even when a corresponding matching calculation exceeded the distance cap. Surfaces connecting sampled epochs are graphical interpolation only.  
TABLE XVIII  
RECORDED COMPUTATIONAL COST AND UNAVAILABLE TIMERS.
<table><tr><td>Component</td><td>Recorded quantity</td></tr><tr><td>Training + monitoring</td><td>2,643.82 s; excludes checkpoint I/O</td></tr><tr><td>Cloud topology computation</td><td>191.48 s; excludes other stages</td></tr><tr><td>Diagram matching wall time</td><td>Not separately recorded</td></tr><tr><td>Nested prediction wall time</td><td>Not separately recorded</td></tr><tr><td>Original ZIP size</td><td>2,192,571,623 bytes</td></tr><tr><td>Compute device</td><td>One Tesla T4; CPU PH/matching</td></tr></table>

Known corrections are explicit: the saved schedule note says 23 instead of 22; all 282 feature-cap skips remain missing; bottleneck analysis is deferred; uncertainty intervals are unadjusted; and the classification-fold comparison is descriptive. Public release and venue-specific length/disclosure requirements remain author decisions rather than invented repository or acceptance claims.

![](images/4052331fc63ae69f0580d97f5272ae4981cf80560181afcfb8cffdeed48e4e74.jpg)

![](images/e5622472b7e074c19fc7a8b2fa4c2628c07030a33e1aab0c4885260ce0d0d6af.jpg)

![](images/08e3a996e5b839271cdaa8761e46f794f517bb4e535eefb3fca2eeb644f09a1f.jpg)

![](images/f33dcf7fa1ed3f304faed4c77de00e40e634011bb07c0f4b573fb841dc857cba.jpg)

![](images/2e33b362970d018501dc1ce180a3d9eda706ab743f582f9af9fb31f7890053f0.jpg)

![](images/6a771050c5e156f10a3df6c760fd2454e6e0ca44d2a67d192c4391c7367ffca2.jpg)

![](images/32bc9a3fcb4645e5f3ec93bbe90d3eb891a35156edeb14a4113815d2da42b5ff.jpg)

![](images/07799ac2300b4f7d719f3d10ed0acdeeb8106f187b778d9f5bbd38624cafe626.jpg)

![](images/274fd4c22e132787848cc12272d132fcc43ec6631425e13dc55e82396298d9c6.jpg)

![](images/8a9541bf3752792551fe92818ace9344eaa9ed605a2dc9d4d113c0762a90503c.jpg)  
Fig. 13. Five-seed mean trajectories of H1 total persistence against mean pairwise cloud distance, for every hidden layer and both representations. Circles mark initialization and squares mark epoch 100; connected intermediate points follow the saved checkpoint schedule. Panel scales vary. Curves are descriptive geometric paths, not a fitted dynamical-system model.

![](images/58bfc1701c4183a83f1df4da2832ef6675eb599bb321abb1b0aec9d5581c7c81.jpg)  
Fig. 14. Exploratory hierarchical clustering of 15 normalized-cloud run signatures. Each of 30 columns is a time-weighted summary (five layers times six metrics). T0/T1: H0/H1 total persistence; E0/E1: lifetime entropy; M1: maximum H1 lifetime; C1: finite H1 count. Columns are standardized across the 15 runs; average-linkage Euclidean clustering determines row and column order. Color saturates at ±2.5 standard deviations.

![](images/08a694f5f1e7be3b6602c875d270e08cbf4cb05919d876ff736cb8fab059d0ef.jpg)  
Fig. 15. Median seed-level Spearman associations between normalized-cloud topology change rates and monitored metric change rates at lag 0. Each cell summarizes five seeds. A small dot marks at least four same-sign seed correlations, without implying statistical significance. T0/T1 denote H0/H1 total persistence; E1 denotes H1 entropy. Shared color limits permit comparisons across supervision conditions.

![](images/f9bef1b19d877bcfe9284b2a3735b18a4d40e64ac610fc92562e443539e62730.jpg)

![](images/ce70f514d61f8e3a9969b31a36f6b006845c3a071e6fabc07434e5eb76becf1e.jpg)

![](images/e5b3a3e94b7d4f805d8806e7403054458b33636f7a47e90c67d48a8c794d94d5.jpg)  
Fig. 16. Within-run temporal separation of six-dimensional normalized-fc3 summary states, averaged over five seeds. Features are H0 total/entropy and H1 total/entropy/maximum/count. Totals, maximum, and count use log(1 + x), followed by feature-wise standardization across all displayed states. Color is Euclidean separation divided by 6. Checkpoints occupy equally spaced matrix cells despite unequal epoch intervals. These are summary-vector separations, not Wasserstein diagram distances.

## REFERENCES

[1] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning requires rethinking generalization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017. [Online]. Available: https: //arxiv.org/abs/1611.03530

[2] D. Arpit et al., “A closer look at memorization in deep networks,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 70, 2017, pp. 233–242. [Online]. Available: https://proceedings.mlr.press/v70/arpit17a.html

[3] C. Lyle, Z. Zheng, E. Nikishin, B. Avila Pires, R. Pascanu, and W. Dabney, “Understanding plasticity in neural networks,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 202, 2023, pp. 23190–23211. [Online]. Available: https://proceedings.mlr.press/v202/lyle23b.html

[4] S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood, and R. S. Sutton, “Loss of plasticity in deep continual learning,” Nature, vol. 632, pp. 768–774, 2024, doi: 10.1038/s41586-024-07711-7.

[5] H. Edelsbrunner, D. Letscher, and A. Zomorodian, “Topological persistence and simplification,” Discrete Comput. Geom., vol. 28, no. 4, pp. 511–533, 2002, doi: 10.1007/s00454-002-2885-2.

[6] P. Bubenik, “Statistical topological data analysis using persistence landscapes,” J. Mach. Learn. Res., vol. 16, no. 3, pp. 77–102, 2015. [Online]. Available: https://jmlr.org/papers/v16/bubenik15a.html

[7] B. Rieck, M. Togninalli, C. Bock, M. Moor, M. Horn, T. Gumbsch, and K. Borgwardt, “Neural persistence: A complexity measure for deep neural networks using algebraic topology,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019. [Online]. Available: https://arxiv.org/abs/1812. 09764

[8] E. Khramtsova, G. Zuccon, X. Wang, and M. Baktashmotlagh, “Convolutional persistence as a remedy to neural model analysis,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), vol. 206, 2023, pp. 10839–10855. [Online]. Available: https://proceedings.mlr.press/v206/khramtsova23a. html

[9] L. Girrbach, A. Christensen, O. Winther, Z. Akata, and A. S. Koepke, “Addressing caveats of neural persistence with deep graph persistence,” Trans. Mach. Learn. Res., 2023. [Online]. Available: https://openreview. net/forum?id=oyfRWeoUJY

[10] G. Naitzat, A. Zhitnikov, and L.-H. Lim, “Topology of deep neural networks,” J. Mach. Learn. Res., vol. 21, no. 184, pp. 1–40, 2020. [Online]. Available: https://jmlr.org/papers/v21/20-345.html

[11] M. W. Mahoney and C. H. Martin, “Traditional and heavy tailed self regularization in neural network models,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 97, 2019, pp. 4284–4293. [Online]. Available: https://proceedings.mlr.press/v97/mahoney19a.html

[12] L. Dinh, R. Pascanu, S. Bengio, and Y. Bengio, “Sharp minima can generalize for deep nets,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 70, 2017, pp. 1019–1028. [Online]. Available: https://proceedings.mlr. press/v70/dinh17b.html

[13] J. Pineau et al., “Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program),” J. Mach. Learn. Res., vol. 22, no. 164, pp. 1–20, 2021. [Online]. Available: https: //jmlr.org/papers/v22/20-303.html

[14] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms,” arXiv:1708.07747, 2017. [Online]. Available: https://arxiv.org/abs/1708. 07747

[15] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Adv. Neural Inf. Process. Syst., vol. 32, 2019, pp. 8024–8035. [Online]. Available: https://proceedings.neurips. cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

[16] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019. [Online]. Available: https://arxiv.org/abs/1711.05101

[17] U. Bauer, “Ripser: Efficient computation of Vietoris–Rips persistence barcodes,” J. Appl. Comput. Topol., vol. 5, no. 3, pp. 391–423, 2021, doi: 10.1007/s41468-021-00071-5.

[18] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer, “Stability of persistence diagrams,” Discrete Comput. Geom., vol. 37, no. 1, pp. 103–120, 2007, doi: 10.1007/s00454-006-1276-5.

[19] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, no. 85, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html

[20] S. Holm, “A simple sequentially rejective multiple test procedure, Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.