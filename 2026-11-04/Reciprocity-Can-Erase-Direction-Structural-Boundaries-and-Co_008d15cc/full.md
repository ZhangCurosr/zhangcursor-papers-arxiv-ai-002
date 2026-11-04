# Reciprocity Can Erase Direction: Structural Boundaries and Controlled Utility of Magnetic Graph Networks

Dawei Han, PhD<sup>1,2,5†</sup>, Zicheng Wang, BS<sup>6†</sup>,

Yaoxing Ren, PhD<sup>1,2†</sup>, Yi Lv, MD<sup>1,2,3,4</sup>,

Rongqian Wu, MD<sup>1,2,3\*</sup>, and Fumin Wang, PhD<sup>1,2,3,4\*</sup>

<sup>1</sup>National Local Joint Engineering Research Center for Precision Surgery & Regenerative Medicine, The First Afiliated Hospital of Xi’an Jiaotong University, Xi’an, PR China

<sup>2</sup>School of Future Technology, Xi’an Jiaotong University, Xi’an, PR China

<sup>3</sup>Med-X Research Institute, The First Afiliated Hospital of Xi’an Jiaotong University, Xi’an, PR China

<sup>4</sup>Department of Hepatobiliary Surgery, The First Afiliated Hospital of Xi’an Jiaotong University, Xi’an, PR China

<sup>5</sup>Ofice of Global promotion, Division of International Cooperation and Exchanges, Xidian University

<sup>6</sup>Arts and Science Department, Cornell University, New York 14850, USA

<sup>†</sup>These authors contributed equally.

## Abstract

Magnetic graph networks encode direction through phase, but a learned graph can erase the intended directional mechanism before prediction. We characterize this failure for supportbased magnetic Laplacians. Exact identities relate imaginary energy and distance from zero phase to amplitude-weighted reciprocity. Top-k selection forces complete collapse when the observed-node count satisfies m ≤ k + 1, and dense support imposes a lower bound on reciprocal-arc fraction. We then construct paired propagation examples with identical node features and symmetric amplitudes but opposite labels, giving a provable chance-level limit for direction-erasing representations. In 72 fixed neural fits, support phase achieved test AUROC 1.0000 on a unilateral ring but 0.5000 on reciprocal ring and complete supports; continuous score phase and a real directed comparator achieved 1.0000 throughout. All models solved the direction-free control. These are oracle-direction mechanism tests, not evidence of learned clinical direction. In the preserved public DeepDRiD development study, continuous phase was active but its robustness diference from zero phase was only +0.0036 (descriptive 95% interval −0.0014 to +0.0095) on validation patients used for checkpoint selection. The contribution is a structural and experimental framework for distinguishing erased direction from unestablished task utility. It provides a precise circumstance in which magnetic phase preserves useful information, without asserting universal or clinical superiority.

Keywords: magnetic graph neural networks; reciprocity; learned graphs; top-k sparsification;   
directional identifiability; controlled benchmarks.

## 1 Introduction

Magnetic graph networks encode edge orientation through complex phases in a Hermitian operator [1]. This is attractive when directional relations are meaningful but an undirected representation loses them. In learned graphs, however, direction is not an intrinsic guarantee: a model first constructs scores, selects neighbors, and converts those selections into an operator. A failure at that conversion can make the intended phase mechanism unavailable before the predictor is optimized.

This paper examines a concrete failure mode. When phase is defined by the diference between binary edge supports, reciprocal selection sets phase to zero even if the two underlying scores difer. Complete top-k support necessarily has this property. Missing observations can therefore turn an otherwise sparse construction into a complete graph without changing the requested k. A nominally magnetic architecture can then operate with exactly its zero-phase Laplacian.

Core claim. Reciprocal top-k support can erase antisymmetric edge information from a support-based magnetic operator. Exact structural diagnostics identify when this occurs; paired directional tasks distinguish that representational failure from a lack of demonstrated benefit on a particular prediction task.

The study is organized around three distinct questions: structural availability, controlled directional utility, and application-specific benefit (Figure 1). Each stage supplies a diferent type of evidence.

We develop this claim through three connected contributions. First, we give exact phase-energy identities and a top-k/node-count boundary, separating support reciprocity from amplitudeweighted reciprocity. Second, we construct paired examples with identical node features and symmetric amplitudes but opposite propagation labels. These give a provable chance-level limit for the direction-erasing representation and an explicit magnetic decoder when orientation survives. A fixed neural experiment tests this mechanism, including a real-valued directed comparator. Third, we reinterpret an audited public retinal-image experiment as a test of task-dependent utility after phase availability has been checked. Its inconclusive comparative gains do not negate the synthetic mechanism and do not establish that images lack learnable direction.

![](images/09a06de2dccd83c9b3340c3b0228e21a083e42347ad69c7939d9c7d807038b65.jpg)  
Figure 1: Research roadmap. a, A complete four-node support illustrates the structural collapse boundary for support-based phase. b, Direction-reversed synthetic pairs share node features and symmetric amplitudes but have opposite propagation labels; s denotes the source and t the target. Direction is supplied in edge scores. $\mathbf { c } ,$ The public DeepDRiD cohort tests application-specific utility. The small interval shows the continuous-minus-zero mean-readout contrast on 100 validation patients used for checkpoint selection: $\Delta R = + 0 . 0 0 3 6$ , descriptive 95% interval $[ - 0 . 0 0 1 4 , + 0 . 0 0 9 5 ]$ . The squares depict the view/region organization, not patient photographs. The arrows summarize the argument, not a transfer of data between experiments.

Our claim concerns representation and experimental attribution, rather than universal accuracy superiority. The magnetic Laplacian and its directed-graph motivation are established [1]; incoming/outgoing real-valued aggregation is also a viable route to directional learning [2]. We do not claim to invent either approach or to be the first to observe that symmetric support has zero phase. The contribution is the quantitative collapse analysis, its interaction with learned top-k graphs and missing observations, and a controlled protocol that makes competing explanations of a null result distinguishable.

## 2 Structural availability of magnetic phase

## 2.1 Operator and assumptions

Consider m observed nodes with no self edges. Let $B \in \{ 0 , 1 \} ^ { m \times m }$ denote selected support and let $W _ { i j } > 0$ exactly when $B _ { i j } = 1$ , with $W _ { i j } = 0$ otherwise. We use $W _ { i j }$ for a message from j to i. Scores may be learned and asymmetric; for example, $W _ { i j } = B _ { i j }$ sigmoid $\left( { { s } _ { i j } } \right)$ . Define

$$
S = \frac { 1 } { 2 } ( { \boldsymbol { W } } + { \boldsymbol { W } } ^ { \top } ) , \quad d _ { i } = \sum _ { j } S _ { i j } , \quad A _ { i j } = \frac { S _ { i j } } { \sqrt { d _ { i } d _ { j } } } , \quad L _ { q } = \Pi - A \odot e ^ { \mathrm { i } 2 \pi q ( B - B ^ { \top } ) } ,\tag{1}
$$

where entries involving zero degrees are zero, $\Pi _ { i i } = \mathbf { 1 } ( d _ { i } > 0 )$ , and exponentiation is entrywise. Strict positivity on selected edges is needed for the support-level equivalences below; under numerical underflow, the energy identities remain valid but a selected zero-weight edge need not contribute.

Let $U = \{ ( i , j ) : B _ { i j } \neq B _ { j i } \}$ be the ordered entries corresponding to unilateral connections. Define

$$
E _ { U } = \sum _ { ( i , j ) \in U } A _ { i j } ^ { 2 } , \qquad E _ { A } = \| A \| _ { F } ^ { 2 } , \qquad r _ { E } = 1 - \frac { E _ { U } } { E _ { A } } \quad ( E _ { A } > 0 ) .\tag{2}
$$

Thus $r _ { E }$ weights reciprocity by normalized amplitude energy. It generally difers from the reciprocal-arc fraction $\begin{array} { r } { r _ { B } = \sum _ { i j } B _ { i j } B _ { j i } / \sum _ { i j } B _ { i j } } \end{array}$

Proposition 1 (Exact phase-energy identities). For the operator in Equation (1),

$$
\Vert \Im L _ { q } \Vert _ { F } ^ { 2 } = \sin ^ { 2 } ( 2 \pi q ) E _ { U } ,\tag{3}
$$

$$
\Vert L _ { q } - L _ { 0 } \Vert _ { F } ^ { 2 } = 4 \sin ^ { 2 } ( \pi q ) E _ { U } .\tag{4}
$$

For $0 < q < 1 / 2 , \Im L _ { q } = 0$ if and only $i f B = B ^ { \top }$ . For $0 < q \leq 1 / 2 , L _ { q } = L _ { 0 }$ if and only if $B = B ^ { \intercal }$ . When $E _ { A } > 0$ , the normalized operator distance is exactly

$$
\frac { \| L _ { q } - L _ { 0 } \| _ { F } } { \| A \| _ { F } } = 2 | \sin ( \pi q ) | \sqrt { 1 - r _ { E } } .\tag{5}
$$

Proof. Reciprocal pairs have zero phase and contribute nothing to either norm. On unilateral entries the phase $\mathrm { i s } \pm 2 \pi q .$ . Squaring the imaginary part yields $A _ { i j } ^ { 2 } \sin ^ { 2 } ( 2 \pi q )$ ; subtracting the zero-phase entry gives $A _ { i j } ^ { 2 } | e ^ { \pm \mathrm { i } 2 \pi q } - 1 | ^ { 2 } = 4 A _ { i j } ^ { 2 } \sin ^ { 2 } ( \pi q )$ . Summing proves the identities. Positive supported amplitudes imply $E _ { U } = 0$ exactly when there are no unilateral pairs. The specified charge ranges and division by $E _ { A }$ give the remaining statements. □

The endpoint $q = 1 / 2$ is consequential: an asymmetric support has zero imaginary part but can yield a real signed operator diferent from $L _ { 0 }$ . Consequently, an imaginary-norm diagnostic alone cannot detect every kind of phase efect. At $q = 0$ , equality to $L _ { 0 }$ is trivial. We call $L _ { q } = L _ { 0 }$ operator collapse; this definition avoids conflating zero imaginary part with complete phase removal.

Proposition 2 (Top-k structural boundary). Suppose each node selects exactly $k _ { \mathrm { e f f } } = \operatorname* { m i n } ( k , m -$ 1) distinct other observed nodes. If $k \geq m - 1$ , then support is complete and $L _ { q } = L _ { 0 }$ for every $q .$ For $1 \leq k _ { \mathrm { e f f } } \leq m - 1$

$$
r _ { B } \geq \operatorname* { m a x } \left\{ 0 , 2 - \frac { m - 1 } { k _ { \mathrm { e f f } } } \right\} .\tag{6}
$$

Proof. Complete support is reciprocal. Otherwise let r be the number of reciprocal unordered pairs and u the number of unilateral arcs. There are m $k _ { \mathrm { e f f } } = 2 r + u$ arcs, while $r + u \le m ( m - 1 ) / 2$ Therefore $u \leq m ( m - 1 - k _ { \mathrm { e f f } } )$ and $u \leq m k _ { \mathrm { e f f } }$ . Substitution in $r _ { B } = 1 - u / ( m k _ { \mathrm { e f f } } )$ proves the bound. □

At $k = 0$ or $m \leq 1$ , the graph has no edges and its operator is zero. For $k < m - 1$ , nonzero phase is possible but not assured: a reciprocal ring with two neighbors already collapses for $m > 3$ . Moreover, Equation (6) controls an unweighted count; it cannot be substituted for $r _ { E }$ in Equation (5) without assumptions about amplitudes. High reciprocal counts can coexist with energy concentrated on a few unilateral edges.

Corollary 3 (Missing observations can force collapse). For a fixed requested k, any observed-node count $2 \leq m \leq k + 1$ forces complete reciprocal support. With four regional nodes per retained photograph and $k = 3$ , a single photograph lies exactly on this boundary.

These statements are pointwise in the constructed graph and thus hold at every learned parameter setting satisfying the assumptions. For a fixed network that uses direction only through $L _ { q } .$ , replacing q by zero on a collapsed graph leaves its prediction unchanged. This is not a claim that separately trained charged and zero-phase models must have identical parameters or predictions after training on other, phase-active graphs.

a Top-k / observed-node boundary  
![](images/dd4d41fce8a97d5f7872cb413acc04bc9433113bd3c49be1736eb527a7b01cea.jpg)

b Exact phase-energy relationship  
![](images/34f423c31bc687369596a283d91ecc32e1bf7b27a0bf10b8c4a0965016b93cdf.jpg)  
Count-based reciprocity and amplitude-energy reciprocity are different quantities.

Figure 2: Analytic structural boundaries. a, The reciprocal-arc fraction is bounded below by max $\{ 0 , 2 -$ $( m - 1 ) / k _ { \mathrm { e f f } } \}$ , with $k _ { \mathrm { e f f } } = \operatorname* { m i n } ( k , m - 1 )$ . Complete reciprocal support occurs on and above the orange boundary, where support phase collapses. The pale region indicates a zero lower bound, not guaranteed asymmetry. b, Exact normalized operator distance for four fixed charges. Its energy-weighted reciprocity $r _ { E }$ difers from the count-based quantity in panel a. These are analytic values, not fitted experimental measurements.

## 2.2 Continuous scores change the collapse condition

A score-based phase uses

$$
\Theta _ { i j } = 2 \pi q \operatorname { t a n h } ( ( s _ { i j } - s _ { j i } ) / \tau ) , \qquad L _ { \Theta } = \Pi - A \odot e ^ { \mathrm { i } \Theta } , \quad \tau > 0 .\tag{7}
$$

Only the selected edge union matters, since A is zero elsewhere. Antisymmetry of Θ preserves Hermitian structure. Directly,

$$
\Vert L _ { \Theta } - L _ { 0 } \Vert _ { F } ^ { 2 } = 4 \sum _ { i j } A _ { i j } ^ { 2 } \sin ^ { 2 } ( \Theta _ { i j } / 2 ) .\tag{8}
$$

Reciprocal support no longer sufices for collapse: unequal reciprocal scores can retain phase even on complete support. This is a change of representation, not a consequence of Proposition 2 for the same operator. The phase path is diferentiable on retained edges; hard top-k selection remains discrete. Nonzero phase or gradient alone does not establish useful prediction, causal direction, or a benefit over real directed aggregation.

## 3 A paired test of directional utility

## 3.1 Identifiability before optimization

We use a controlled task where direction has known predictive content. On an eight-node ring, mark a source and one of its two adjacent nodes as the target. A pulse follows a ground-truth directed cycle for one step. The binary label states whether it reaches the target. Each example is paired with the reverse cycle, using the same features, source, target, and node permutation; its label is reversed. The direction is supplied through edge scores, not inferred from identical node features.

Scores are 2 along the ground-truth cycle, 1 along the reverse edges, and −2 on other ofdiagonal pairs. Diagonal entries are excluded. Row-wise top-k selection followed by sigmoid weighting produces W. For $k = 1$ , only the cycle is selected; for $k = 2 .$ , both ring directions are selected; for $k = 7$ , support is complete. Each twin pair has the same S for all three settings. $\mathrm { A t } ~ k = 2$ and $7 , W$ remains asymmetric although support is reciprocal. At these settings the ground-truth cycle is the higher-score direction; the weaker reverse edges are part of the observed weighted graph, not the ground-truth one-step propagation process.

Proposition 4 (Paired representation gap). On a balanced set of paired opposite-label examples with identical $( X , S )$ , every deterministic predictor that depends only on $( X , L _ { 0 } )$ produces equal scores within each pair. Its empirical AUROC and threshold accuracy are exactly $1 / 2$ , and its mean binary cross-entropy is at least log 2. The same conclusion holds for support-phase predictors on the reciprocal $k = 2$ and $k = 7$ constructions. With $q = 1 / 4$ , the support-phase operator at $k = 1$ , and the continuous-phase operator at all three k, admit a one-step decoder with perfect accuracy on this construction.

Proof. The direction-erasing representation assigns a common score to each pair. The positive and negative empirical score multisets are therefore identical, implying AUROC $1 / 2$ , including half credit for ties. Any fixed threshold yields one correct label per pair. Pairwise cross-entropy $\begin{array} { r } { \mathrm { i s } - \frac { 1 } { 2 } \log ( p ( 1 - p ) ) \geq \log 2 } \end{array}$ . Support collapse gives the same representation for reciprocal support. For the constructive part, let $e _ { s }$ be the source pulse. The sign of $( - \Im L e _ { s } ) _ { t }$ is positive when the target is the successor and negative for the reverse direction: the selected normalized amplitude is positive and the relevant phase lies strictly between 0 and $\pi .$ . The target indicator locates t independently of node order. Thresholding this sign recovers the label. □

The proposition describes a representational possibility, not a guarantee of gradient-based optimization. We therefore also fit small neural predictors. A direction-free control uses the sign of an observed target feature as the label; twins share that label. This control checks that collapse does not imply inability to learn any task. It is deliberately simple and is not evidence about the complexity of natural tasks.

## 3.2 Locked neural protocol

Each node has four features: source and target indicators and two noise channels. The target value of one noise channel has a sign that alternates deterministically with pair index, ensuring balance, and a random magnitude uniform on [0.5, 1.5); it determines only the direction-free label. Other noise values are standard normal. Node permutations are sampled independently across pairs. Pair IDs are used only for auditing and never as model inputs. There are 256 training, 64 validation, and 128 test pairs (512, 128, and 256 graphs) per task. Twin members remain within the same split, and split generators are independent. The two tasks share the graph/feature realizations and difer in labels, enabling a direct task control.

All four models use a width-16 linear/ReLU node encoder and a target-conditioned readout with a $3 2  1 6  1$ head. Magnetic models use one order-1 complex Chebyshev layer with a complex residual and separate real/imaginary ${ \mathrm { R e L U } } ;$ real and imaginary target embeddings are concatenated. The zero-phase control has the same complex parameters and architecture. The real directed model uses separate normalized incoming/outgoing messages and one residual layer, concatenating a zero imaginary block at readout. Parameter counts and their limits are reported with the results; matching nominal readout dimensions does not equalize efective capacity.

The four fixed operators are zero phase, support phase $( q = 0 . 2 5 )$ , continuous phase $( q = 0 . 2 5$ $\tau = 1 )$ , and real directed aggregation. All receive the same scored graph before their declared conversion; no operator is given labels. Each task/operator/k combination uses three initialization seeds, totaling 72 fits. Each fit runs 30 epochs of AdamW with learning rate 0.003, weight decay 0.0001, batch size 64, and gradient clipping at norm 5. The earliest minimum validation BCE selects the saved checkpoint. No charge search, test-directed tuning, or seed exclusion is performed. Test AUROC and accuracy are reported per fit and summarized by the mean and range across seeds, without treating seed variation as population uncertainty.

This is an oracle-direction positive control. Because twins have identical X, a deterministic relation builder using X alone could not recover their opposite orientations. Success with supplied scores demonstrates that an operator can retain and use a known directional signal; it does not demonstrate that a learned retinal graph recovers such a signal.

## 4 Synthetic results: a structural failure and its remedy

All 72 planned fits completed, yielding 18,432 test predictions across runs. Each task contains only 128 unique test pairs; repeated seeds and models do not multiply its sample size. All saved checkpoints reproduced their selected validation BCE and all saved test probabilities exactly under replay. An independent pairwise-ranking implementation verified all 72 AUROCs.

On the directional task, support phase with $k = 1$ achieved AUROC and accuracy of 1.0000 in every seed, while the matched zero-phase model achieved 0.5000. With reciprocal support at $k = 2$ and 7, the support-phase model fell to chance, as predicted by Proposition 4. Continuous phase achieved AUROC and accuracy of 1.0000 at all three settings. The real directed comparator also achieved 1.0000 throughout. Thus, the demonstrated advantage is relative to direction-erasing representations; these observations do not establish superiority over real directed learning.

All models achieved AUROC and accuracy of 1.0000 on the direction-free feature control, including collapsed support-phase and zero-phase models. Collapse therefore removes the ability to distinguish the designed orientations, rather than rendering the network generally untrainable.

Magnetic and zero-phase models each had 2,161 trainable parameters; the real directed comparator had 2,801. The exact chance-level argument is independent of parameter count for predictors restricted to the same erased representation. We do not infer computational eficiency or optimal capacity from these small implementations.

Table 1: Synthetic test AUROC, averaged across all three initialization seeds. Top rows: directiondependent propagation. Bottom rows: direction-free feature control. Test graphs are held out from training and checkpoint selection. All seed-level results and accuracies appear in the appendix.
<table><tr><td>k / task</td><td>Zero phase</td><td>Support phase</td><td>Continuous</td><td>Real directed</td></tr><tr><td>1</td><td>0.5000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>2</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>7</td><td>0.5000</td><td>0.5000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>1 (feature)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>2 (feature)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>7 (feature)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr></table>

Two raw float32 runs (zero and support phase at $k = 7 ,$ , seed 20260922) had AUROC 0.5000305 instead of exactly 0.5. Their twin inputs and Laplacians were bitwise identical, but a single twin pair difered by $5 . 9 6 \times 1 0 ^ { - 8 }$ in predicted probability due to floating-point evaluation; accuracy remained 0.5. Averaging scores within pairs restored AUROC 0.5 as a diagnostic. All reported metrics use the original probabilities, and the complete table retains five decimal places. This numerical efect does not represent recovered directional information.

Operator measurements agreed with the mechanism (Figure 4). Support reciprocity and energy reciprocity were both zero at $k = 1$ and one at $k = 2$ and 7. Normalized support-phase distance from $L _ { 0 }$ was ${ \sqrt { 2 } } , 0$ , and 0, respectively. Continuous-phase distance was 1.1262, 1.1262, and 1.0966. The maximum discrepancy in the squared-norm identity was $3 . 5 5 \times 1 0 ^ { - 1 5 }$ . The explicit one-step sign decoder classified every held-out pair correctly for continuous phase at every k.

![](images/814cd98df3ab9ab008b5c9c84e77387c3d4acdd31c450d1a3a5fc41224e1e1cc.jpg)  
Test AUROC: mean of 3 seeds | 128 held-out pairs per task | supplied directional scores

Figure 3: Controlled directional utility. a, An illustrative source–target pair on an eight-node cycle. Reversal changes whether the source pulse reaches the adjacent target in one step, while preserving node features and symmetric amplitudes. $\mathbf { b , c , }$ Complete test AUROC matrices for the propagation task and direction-free feature control. Each cell is the mean of three initialization seeds on 128 held-out pairs; all 72 fits are retained. Columns correspond to unilateral $( k = 1 )$ , reciprocal-ring $\left( k = 2 \right)$ , and complete $( k = 7 )$ support. Color encodes $\mathrm { A U R O C }$ , not model identity. Values are rounded to four decimals; unrounded per-seed values, including the small float32 departures from 0.5, are retained in the source data and appendix. Directional scores are supplied inputs.

b Measured phase retention  
![](images/f71112867751232ae6c23b2503aac602843165bbcc15cca49bc934703a721e7f.jpg)

![](images/20ce44bc5e658c647aba0e43cbf12fac05b47bce88ed8803eaeb9b681fa4f06b.jpg)  
Figure 4: How continuous phase preserves reciprocal score information. ${ \mathbf { a } } ,$ An illustrative pair has $s _ { i j } = 2$ and $s _ { j i } = 1$ with reciprocal binary support. At $q = 0 . 2 5$ and $\tau = 1$ , support phase is zero whereas score-based phase is $\theta _ { i j } = 1$ .1963 radians; the reverse entry has the opposite phase. b, Measured operator distance on the paired benchmark, normalized by $\| A \| _ { F }$ . Reciprocal support erases support phase at $k = 2$ and $^ { 7 , }$ while continuous phase retains the asymmetry. Numeric labels are rounded; plotted values use the full recorded precision. Nonzero distance is a structural property, and the separate propagation experiment establishes utility on its designed task.

## 5 Real-data test: availability does not establish utility

## 5.1 Audited public cohort and development design

The real-data component uses the public DeepDRiD repository [3], revision 56d8af7, under CC BY-SA 4.0. Four training patients with ambiguous filename/eye correspondence were excluded before fitting, leaving 496 patients and 1,984 photographs: 296 training, 100 validation, and 100 test patients. The endpoint is patient-level maximum gradable-eye DR severity of at least grade 2; it does not include a diabetic macular edema criterion. There are four photographs per patient. No patient-level clinical covariates, omics, or follow-up outcomes are included.

Frozen ImageNet ResNet-18 features [4] provide four regional nodes per photograph, at most 16 per patient. A width-128 query/key relation builder selects $k = 2$ neighbors. Magnetic models use two order-2 complex layers. A preceding $k = 3$ diagnostic on single-view, four-node graphs observed reciprocal support, zero imaginary operator norm, and unchanged predictions when charge was removed. This is the exact boundary of Proposition 2, rather than evidence that the endpoint lacks directional content. Reducing k to 2 removed forced collapse in that diagnostic, but did not itself establish accuracy benefit.

The preserved development extension screened $q \in \{ 0 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 2 5 \}$ within three stratified folds of the 296 training patients (15 fits), selecting $q = 0 . 2 5$ by mean best balanced BCE. Five model types (support, continuous, zero, shufled continuous, and real directed) were then fitted with mean and relational readouts and three seeds (30 fits). Training used 30 epochs, AdamW with learning rate 0.001 and weight decay 0.0001, batch size 16, gradient clipping at 5, and common patient-indexed schedules retaining one to four views. Validation BCE averaged equally over retained-view counts selected the earliest minimum checkpoint. No new model in this extension was evaluated on the old test partition.

All 15 nonempty view subsets of the 100 validation patients were scored. The development measure R is AUROC averaged first across subsets of the same size and then equally across one-, two-, and three-view conditions. Probabilities from three seeds are averaged before calculating this measure. Missing-view labels remain the original patient-level labels. The same validation patients were used for checkpoint selection and evaluation; these are selected development estimates, not independent generalization estimates. The older benchmark had already examined the original test set and is retained as exploratory historical evidence, not repurposed as a new confirmatory test.

## 5.2 What the observations support

Table 2: Preserved DeepDRiD development results on 100 validation patients used for checkpoint selection. Each estimate uses a three-seed probability ensemble. These are not independent test results.
<table><tr><td></td><td colspan="2">R</td><td colspan="2">Full-view AUROC</td></tr><tr><td>Model</td><td>Mean</td><td>Relational</td><td>Mean</td><td>Relational</td></tr><tr><td>Support phase</td><td>0.9351</td><td>0.9367</td><td>0.9504</td><td>0.9548</td></tr><tr><td>Continuous phase</td><td>0.9403</td><td>0.9399</td><td>0.9568</td><td>0.9540</td></tr><tr><td>Zero phase</td><td>0.9367</td><td>0.9367</td><td>0.9552</td><td>0.9508</td></tr><tr><td>Shuffled phase</td><td>0.9387</td><td>0.9356</td><td>0.9512</td><td>0.9532</td></tr><tr><td>Real directed</td><td>0.9356</td><td>0.9342</td><td>0.9540</td><td>0.9488</td></tr></table>

With mean readout, continuous-phase R was 0.9403 versus 0.9367 for zero phase and 0.9356 for real directed aggregation. The continuous-minus-zero diference was +0.0036, with a descriptive 95% patient-bootstrap interval from $- 0 . 0 0 1 4 \mathrm { t o } + 0 . 0 0 9 5$ the continuous-minus-directed diference was $+ 0 . 0 0 4 7 \ ( - 0 . 0 0 3 0 \ \mathrm { t o \ + 0 . 0 1 2 7 } )$ . Both include zero. Continuous minus shufled phase was $+ 0 . 0 0 1 6 \ \left( - 0 . 0 0 4 4 \ \mathrm { t o } \ + 0 . 0 0 7 2 \right)$ Continuous minus support phase was +0.0052 (+0.0006 to +0.0110), but the interval is conditional on selected models and is unadjusted for selection and multiplicity. It does not establish independent superiority.

The relational readout did not strengthen the main attribution: continuous-minus-zero R $\mathrm { w a s + 0 . 0 0 3 2 ~ ( - 0 . 0 0 1 3 ~ t o ~ + 0 . 0 0 9 3 ) }$ , and the phase-by-readout interaction relative to zero phase $\mathrm { w a s \ - 0 . 0 0 0 4 \ ( - 0 . 0 0 6 7 ~ t o ~ + 0 . 0 0 6 8 ) }$ . The intervals use 2,000 class-stratified patient bootstrap draws shared across models and subsets; repeated subset predictions do not increase the number of independent patients.

Continuous phase was mechanically active. Full-view mean-readout checkpoints had mean $\lVert \Im L \rVert _ { F } / \lVert L \rVert _ { F } = 0 . 3 4 9 4$ across seeds, with finite nonzero phase-only query gradients. This denominator difers from the normalized amplitude norm in Equation (5). These diagnostics show an available phase pathway, not that its direction is clinically meaningful. Independent replay previously reproduced all 45 selected checkpoints and all 45,000 final validation probabilities.

DeepDRiD therefore supplies the third part of the argument: removing structural collapse is insuficient to establish task utility. The observations remain compatible with weak direction, an inefective learned relation builder, readout/optimization limitations, limited sample size, or no material gain for this endpoint. They neither prove absence of direction in static images nor furnish a counterexample to every magnetic architecture.

## 6 Discussion

## 6.1 Meaning of the claim

The practical contribution is a sequence of questions with diferent evidential requirements. First, can the constructed operator distinguish the relevant orientations? The exact collapse identities answer a structural part of this question without training. Second, can the implementation use a

![](images/0f9ff4dd7edea5e6fea91b465782cd7c864f3b0b7d992934add0a2ee5b3bf643.jpg)

![](images/715b43aa6ba7a3d0fc0c04d2bd1b1fdda85a762c85241d92753f9add6d345441.jpg)  
100 validation patients; checkpoint selection and evaluation used the same patients.  
95% patient-bootstrap intervals are descriptive and unadjusted for selection or multiplicity.

Figure 5: Preserved DeepDRiD development contrasts. $^ { \mathbf { a } , \mathbf { b } }$ , Continuous-phase minus comparator diferences under mean and relational readouts, respectively. Colors and marker shapes identify the comparator consistently; both panels use the same horizontal scale. Points are three-seed ensemble estimates, with descriptive 95% paired patient-bootstrap intervals; the dashed line is zero diference. All eight contrasts are shown. Checkpoint selection and evaluation used the same 100 validation patients, and intervals are unadjusted for selection or multiplicity. These selected development results do not establish independent clinical superiority.

directional signal when one is supplied? Paired opposite-label examples provide an identifiable positive control and expose chance-level limits of the erased representation. Third, does the learned direction improve the target application? That requires controlled comparisons on appropriate independent data. These questions should not be answered by a single end-to-end AUROC.

The advantage established here is specific: phase can preserve orientation that a symmetrized representation discards, and score-based phase can avoid the loss caused by reciprocal binary support. This supplies a reason to use magnetic encoding on a suitable task. It does not supply a reason to prefer it to every real-valued directed model. The directed comparator tests that distinction explicitly. Nor is complex arithmetic alone responsible: the zero-phase control retains complex filters, separating operator direction from nominal complex capacity.

## 6.2 Design implications

A fixed neighbor budget should be audited against the distribution of observed node counts, especially under missing modalities or views. Reporting only the configured k hides completesupport cases with $m \leq k + 1$ . Reporting $r _ { B }$ alone can also be misleading; $r _ { E }$ and $\| L _ { q } - L _ { 0 } \| _ { F }$ quantify the actual retained phase energy. Charge endpoints require care, since $q = 1 / 2$ has a zero imaginary part without necessarily equaling $L _ { 0 }$ . These diagnostics can identify a nonfunctional ablation before expensive training.

Continuous scores provide one way to preserve reciprocal directional diferences. Their success depends on what scores encode. On our paired task scores reveal the ground-truth orientation; on the retinal task they are learned from images without directional supervision. Bridging that gap requires tasks with independently justified directional relations, such as an observed ordering or a measured flow, followed by evaluation that does not reuse selection data. A nonzero phase on an arbitrary similarity graph is not itself a meaningful scientific mechanism.

## 6.3 Limits and remaining evidence

The exact identities follow directly from the chosen operator and should be judged as useful diagnostics, not a broad new theory of all magnetic GNNs. The synthetic benchmark has one small ring family, supplied edge scores, a local propagation label, fixed charges, and a simple direction-free control. Perfect separation is possible by construction. It validates the stated mechanism rather than realistic noise tolerance, multi-hop expressivity, large-graph scalability, or learned direction recovery. Seed ranges do not estimate population uncertainty. The models are controlled implementations rather than exhaustive tuned competitors; efective capacity difers between real and complex variants.

The clinical component is small, retrospective, image-only, and subject to development selection and prior test access. It has no untouched external cohort and supplies no clinical superiority claim. Moreover, $L _ { q } \neq L _ { 0 }$ is only an operator distinction. A particular input or readout can ignore it, and nonzero local phases need not imply nontrivial cycle flux or a distinct spectrum under every gauge transformation. We do not infer utility from spectral change or claim full-network gauge invariance.

## 7 Conclusion

Reciprocity can suppress a magnetic model’s intended directional mechanism before learning. Exact energy identities and the top-k/observed-node boundary make that suppression measurable. Paired propagation controls establish where retaining direction can help, while the DeepDRiD comparison shows why structural availability must be followed by application-specific validation. The central claim is therefore a testable boundary of representation and utility, with a concrete remedy for score information lost by reciprocal support.

## Declarations

Data and code availability. DeepDRiD is available from its authors’ public repository [3]. The project retains the source revision and license, download/audit tools, image and manifest fingerprints, frozen-feature provenance, all configurations, selected checkpoints and histories, patient-level predictions, and scripts for the reported figures and tables. Data retain the source CC BY-SA 4.0 license. A permanent public code archive and DOI have not yet been assigned.

Ethics and consent. This is a secondary analysis of the released public dataset with no new recruitment or clinical intervention. Original collection and consent arrangements are described by the dataset authors. No new institutional approval or exemption is claimed; the submitting institution must determine and document applicable requirements.

Authorship, funding, and competing interests. Dawei Han, Zicheng Wang, and Yaoxing Ren contributed equally. Rongqian Wu and Fumin Wang are corresponding authors. Detailed author contributions, funding, and competing-interest declarations require confirmation by the responsible investigators before submission. No funding award or absence of competing interests is asserted on their behalf.

AI assistance. AI tools assisted with implementation, analysis scripting, numerical checks, and manuscript drafting. The responsible authors must review the evidence, interpretation, and applicable journal disclosure requirements. This research draft is not a clinical decision tool or a finalized submission.

## References

[1] Xitong Zhang, Yixuan He, Nathan Brugnone, Michael Perlmutter, and Matthew Hirn. MagNet: A neural network for directed graphs. In Advances in Neural Information Processing Systems, volume 34, pages 27003–27015, 2021. URL https://proceedings.neurips.cc/ paper\_files/paper/2021/hash/e32084632d369461572832e6582aac36-Abstract.html.

[2] Emanuele Rossi, Bertrand Charpentier, Francesco Di Giovanni, Fabrizio Frasca, Stephan Günnemann, and Michael M. Bronstein. Edge directionality improves learning on heterophilic graphs. In Proceedings of the Second Learning on Graphs Conference, volume 231 of Proceedings of Machine Learning Research, pages 25:1–25:27, 2024. URL https://proceedings.mlr.press/v231/rossi24a.html.

[3] Ruhan Liu, Xiangning Wang, Qiang Wu, Ling Dai, Xi Fang, Tao Yan, Jaemin Son, Shiqi Tang, Jiang Li, Zijian Gao, et al. DeepDRiD: Diabetic retinopathy—grading and image quality estimation challenge. Patterns, page 100512, 2022. doi: 10.1016/j.patter.2022.100512. URL https://github.com/deepdrdoc/DeepDRiD. Public repository, CC BY-SA 4.0; revision 56d8af71319803150d0e7b90533e0918c7bf85ee.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. URL https://openaccess.thecvf.com/content\_cvpr\_ 2016/html/He\_Deep\_Residual\_Learning\_CVPR\_2016\_paper.html.

## A Reproducibility and complete results

The synthetic protocol is stored in configs/direction\_v07.json; the generator and model are in src/magnn\_retina/direction\_benchmark.py. The training and independent replay/report entry points are analysis/train\_direction.py and analysis/report\_direction.py. Results include generated tensor splits, source/dependency identity, all selected checkpoints and epoch histories, per-graph test probabilities, all 72 run summaries, independent verification, and vector figures. Dataset generation seeds for train, validation, and test are 2026092101, 2026092102, and 2026092103. Initialization seeds are 20260921, 20260922, and 20260923. The complete test grid is shown below; no cell or seed has been excluded.

Table 3: Complete synthetic test results. A21–A23 are AUROC and C21–C23 are accuracy for seeds 20260921–20260923. AUROC is shown to five decimals to expose the small finite-precision deviation described in the main text.
<table><tr><td>Task</td><td></td><td>k Model</td><td>A21</td><td>A22</td><td>A23</td><td>C21</td><td>C22</td><td>C23</td></tr><tr><td>Direction</td><td>1</td><td>Zero phase</td><td>0.50000</td><td>0.50000</td><td>0.50000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direction</td><td>1</td><td>Support phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>1</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>1</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>2</td><td>Zero phase</td><td>0.50000</td><td>0.50000</td><td>0.50000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direction</td><td>2</td><td>Support phase</td><td>0.50000</td><td>0.50000</td><td>0.50000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direction</td><td>2</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>2</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>7</td><td>Zero phase</td><td>0.50000</td><td>0.50003</td><td>0.50000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direction</td><td>7</td><td>Support phase</td><td>0.50000</td><td>0.50003</td><td>0.50000</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr><tr><td>Direction</td><td>7</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Direction</td><td>7</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>1</td><td>Zero phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>1</td><td>Support phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>1</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>11</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>2</td><td>Zero phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>2</td><td>Support phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>2</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>2</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>7</td><td>Zero phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>7</td><td>Support phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>7</td><td>Continuous phase</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Feature</td><td>7</td><td>Real directed</td><td>1.00000</td><td>1.00000</td><td>1.00000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr></table>

## B Additional real-data implementation details

The retrospective retinal fits remain those of version 0.6; version 0.7 adds no patient observations or new retinal fitting. The charge screen used seed 20260920 in each of three training-patient folds. Its mean best balanced BCE values were 0.3827, 0.3771, 0.3777, 0.3755, and 0.3748 for ascending charges; these are tuning statistics. Final training seeds were 20260920–20260922. The shufledphase control permutes upper-triangle phase values within the observed edge union, restores antisymmetry, and uses a deterministic patient/seed permutation without label information. The control is retrained, so independently learned amplitudes can difer.

The relational readout concatenates mean real features, mean imaginary features, mean magnitude, and a per-channel phase-aligned edge correlation. For $u _ { i c } = z _ { i c } / \operatorname* { m a x } ( | z _ { i c } | , 1 0 ^ { - 8 } )$ and $\begin{array} { r } { \widetilde { S } = S / \sum _ { i j } S _ { i j } } \end{array}$ , the last statistic is $\begin{array} { r } { \sum _ { i j } \widetilde { S } _ { i j } \Re \{ \overline { { u } } _ { i c } e ^ { \mathrm { i } \Theta _ { i j } } u _ { j c } \} } \end{array}$ . Isolated-graph edge statistics are zero. The real directed control uses zero imaginary features and zero phase for this statistic. Magnetic models have 428,673 trainable parameters and real directed models 430,721, excluding the common frozen backbone. Equal nominal readout layout does not imply equal active head columns.

The source license and revision, quarantined-patient audit, patient partitions, frozen-feature provenance, original v0.5 and v0.6 reports, and prior prediction files are retained. The 100 validation patients comprise 50 negative and 50 positive cases. Quarantined training patient IDs were 56, 77, 164, and 167. Their removal was based on metadata ambiguity rather than outcomes or predictive performance. Source repository labels and licensing are not substitutes for institution-specific submission and ethics review.

## C Interpreting related but diferent diagnostics

For fixed $0 < q < 1 / 2$ and positive selected weights, three conditions are equivalent for support phase: reciprocal support, zero imaginary operator, and equality to $L _ { 0 }$ . At $q = 1 / 2$ only the first and third remain equivalent. At $q = 0$ all graphs have equality to $L _ { 0 }$ regardless of support. At empty support $E _ { A } = 0$ , both operators are zero but normalized reciprocity and distance ratios are undefined. These cases should be handled explicitly rather than assigned a misleading ratio.

For a continuous phase, reciprocity of binary support is not a collapse criterion. The relevant quantities are the retained antisymmetric score diferences and the weighted sine terms in Equation (8). For either construction, an operator norm does not quantify a trained predictor’s accuracy, and a small norm alone gives no bound on its output without assumptions about filter and readout norms. A missing-view experiment changes both the graph and available visual information; it cannot isolate a causal clinical mechanism.