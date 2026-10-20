# Physics-anchored residual quantum neural networks for cloud-cover parameterization

Zhuo Wang, Ce Yu, Bin Yang\*

Tianjin University

## Abstract

Compact quantum neural networks have been explored for cloud-cover parameterization, but their predictive value depends on how physical knowledge is incorporated and which classical models provide the reference. We investigate an ofline residual formulation that retains a fitted Xu–Randall cloud-cover estimate and learns a correction to its logit in transformed target space. Using released ICON/DYAMOND-derived data, we compare ten model variants in six- and eight-input settings, with three matched replicates and 10,000 training samples per replicate. All quantum circuits use local Z readout and noiseless exact expectation values. Under a common 30-epoch protocol, residual prediction reduced physical-space test mean squared error relative to direct QNN prediction by 54.3% and 38.8% for six-input XYZ and ZZXY circuits, and by 49.2% and 38.9% for their eight-input versions. Residual QNNs also outperformed the selected compact residual multilayer perceptron in every paired replicate. A larger, longer-trained Grundner-Q1 adaptation nevertheless achieved lower error: residual Q1 reached mean MSEs of 0.00778 and 0.00791, compared with 0.01124 and 0.01082 for residual XYZ. The results support residual prediction for compact simulated QNNs within the tested budget and establish a clear boundary set by stronger classical references. They do not establish computational quantum advantage or online climate-model skill.

## 1 Introduction

## 1.1 Cloud-cover parameterization and existing learning approaches

Cloud-cover parameterizations estimate the cloudy fraction of a model grid cell from resolved atmospheric variables. This diagnostic connects unresolved cloud structure to the radiation and microphysics represented in a climate model. Coarse spatial resolution makes the relationship dificult: similar mean atmospheric states can contain diferent sub-grid distributions of water and cloud. Storm-resolving simulations provide training examples for learning this relationship, although accurate ofline predictions do not by themselves establish skill after coupling to atmospheric dynamics [1–3].

Semiempirical schemes such as Xu–Randall express cloudiness through a compact physical relationship [4]. Classical neural networks ofer more flexible mappings, with Grundner et al. demonstrating learned cloud-cover parameterizations from coarse-grained ICON simulations [1]. Related work on unresolved atmospheric processes and scientific machine learning shows how data-driven models can be combined with existing process knowledge [5–7]. One consequential choice is whether a network learns the complete diagnostic or corrects a useful approximation. The latter retains information from the approximation and its calibration, changing what the trainable component must represent.

Quantum neural networks (QNNs) provide another compact parameterized function class. Classical inputs enter a circuit as rotation angles, and measured expectation values supply regression features. Repeated encoding, or data re-uploading, changes the accessible input-dependent frequency structure [8, 9]. Pastori et al. applied such circuits to cloud-cover prediction and compared them with classical networks of similar parameter counts [10]. Their data, transformations, circuits, and fitted physical baseline define our starting point. Whether a circuit is useful for prediction remains an empirical question requiring appropriate classical references. Precise simulations on classical computers do not prove quantum computing superiority [11, 12].

## 1.2 Research questions and contributions

We ask whether a compact QNN benefits from correcting a physical cloud-cover estimate rather than predicting cloud cover directly. A second question is whether any advantage persists when classical models receive the same physical anchor, and when a stronger published classical architecture is included. Finally, we examine whether the findings persist in the paper-defined eight-input configuration on the same sample splits.

We address these questions with a bounded correction in transformed-logit space, direct and residual XYZ and ZZXY circuits, compact classical controls, and direct and residual Grundner-Q1 adaptations. The contribution is an evaluated prediction design and a matched comparison with explicit limits. Direct and residual circuits have identical trainable parameter counts; all residual controls receive the same inherited baseline. Six- and eight-input comparisons share sample manifests. This design preserves the distinction between near-sized comparisons under a shared training protocol and stronger classical references trained under a diferent protocol.

## 2 Methods

## 2.1 Xu–Randall anchoring and residual prediction

Let x denote a cell’s atmospheric inputs and $c \in [ 0 , 1 ]$ its physical cloud fraction. We use the fitted Xu–Randall-derived diagnostic supplied by the reference implementation [4, 10]:

$$
c _ { \mathrm { X R } } ( x ) = \operatorname* { m i n } \Bigl \{ 1 , \Bigl [ 1 - e ^ { - \alpha \left( q _ { l } + q _ { i } \right) } \Bigr ] \mathrm { R H } ^ { \beta } \Bigr \} , \quad \alpha = 4 . 0 3 4 \times 1 0 ^ { 4 } , \quad \beta = 0 . 9 9 4 2 .\tag{1}
$$

Here $q _ { l }$ and $q _ { i }$ are liquid-water and ice specific contents, and relative humidity is diagnosed from specific humidity, temperature, and pressure. The two constants remain fixed. Their earlier calibration supplies information beyond the selected training subset; we report them separately from newly trainable parameters. Equation (1) records the implementation used here, rather than a new fit of the complete original 1996 scheme.

With the inherited target transform $z = \mathcal { T } ( c )$ , define

$$
a ( x ) = \mathrm { c l i p } ( \mathcal { T } ( c _ { \mathrm { X R } } ( x ) ) , \epsilon , 1 - \epsilon ) , \qquad \epsilon = 1 0 ^ { - 3 } .\tag{2}
$$

A trainable correction $r _ { \theta } ( x )$ produces

$$
\hat { z } ( x ) = \sigma [ \mathrm { l o g i t } a ( x ) + r _ { \theta } ( x ) ] , \qquad \hat { c } ( x ) = { \mathcal T } ^ { - 1 } ( \hat { z } ( x ) ) ,\tag{3}
$$

where $\sigma ( s ) = ( 1 + e ^ { - s } ) ^ { - 1 }$ . The correction is additive in the logit of the transformed target, rather than in physical cloud fraction. Figure 1 shows the fixed physical branch and trainable quantum branch.

Every residual output head starts at zero, recovering $a ( x )$ up to numerical precision while retaining random circuit or hidden-layer initialization. Clipping keeps the anchor logit finite at the endpoints. We minimize $N ^ { - 1 } \sum _ { i } [ \hat { z } _ { i } - \mathcal { T } ( c _ { i } ) ] ^ { 2 }$ and evaluate after inverse transformation. Observed targets are never converted to logits, so zero and one remain valid finite training targets. The sigmoid bounds predictions without constraining the correction to be small; the transformed loss weights physical errors diferently from physical-space MSE.

## 2.2 Quantum circuits and local Z readout

We reuse the XYZ and ZZXY implementations with one qubit per input and noiseless exact expectations. Each input is repeatedly encoded through an $R _ { X }$ rotation. XYZ encoding blocks contain nearest-neighbor Ising ZZ, XX, and YY operations; its subsequent variational blocks additionally apply local $R _ { X }$ rotations. ZZXY encoding blocks combine Ising ZZ interactions with local $R _ { Y }$ rotations, followed by variational blocks containing local $R _ { X }$ , Ising ZZ, and local $R _ { Y }$ operations. Encoding and variational layer counts $( n _ { \mathrm { e n c } } , n _ { \mathrm { v a r } } )$ are (4, 2) and (2, 5) for six-input XYZ and ZZXY, and (5, 3) and (2, 7) for eight inputs.

![](images/5361477a5b180f6a19dfb8ffacf9de8204443ca94d790320ef3b4b6d2085a979.jpg)  
Figure 1: Physics-anchored residual QNN. The fixed branch supplies logit $a ( x )$ . The quantum branch supplies local expectations $m _ { i } = \langle Z _ { i } \rangle$ to one afine head. Its zero initialization recovers the clipped anchor. The branches combine before the sigmoid, and the inverse target transform returns physical cloud cover. Training loss is evaluated before that inverse transform; the dashed arrow updates only trainable parameters. Overlapping blocks and representative wires abbreviate the six- or eight-qubit circuit, not its actual layer count. The figure’s T and $T _ { x }$ denote the target and input transforms.

Both prediction modes use the same local observables and afine readout:

$$
m _ { i } ( x ; \vartheta ) = \langle 0 | U _ { \vartheta } ^ { \dagger } ( x ) Z _ { i } U _ { \vartheta } ( x ) | 0 \rangle , \qquad h _ { \theta } ( x ) = b + \sum _ { i = 1 } ^ { d } w _ { i } m _ { i } ( x ; \vartheta ) .\tag{4}
$$

The direct model uses $h _ { \theta }$ to estimate $z ;$ the residual model uses it as $r _ { \theta }$ in $\mathrm { E q . \ ( 3 ) }$ . Local Z readout concerns measured observables and does not remove the circuit’s two-qubit operations. Circuit angles and afine coeficients give 109/114 trainable parameters for six-input XYZ/ZZXY and 201/200 for eight inputs, unchanged between direct and residual versions. These counts describe the regression model, not the memory or execution cost of its state-vector simulator.

## 2.3 Classical residual controls and the Grundner Q1 adaptation

The compact direct multilayer perceptron (MLP) follows the reference structures $6  8  3  7  1$ and $8 \to 1 2 \to 6 \to 2 \to 1$ , with tanh hidden layers and linear outputs. Its residual counterpart preserves the hidden layers and supplies Eq. (3) through a zero-initialized final layer. A linear residual control uses $\boldsymbol { r } _ { \phi } ( \boldsymbol { x } ) = \boldsymbol { w } ^ { \top } \boldsymbol { x } + \boldsymbol { b }$ , with all coeficients initially zero. These models test whether simple classical corrections share the benefit and provide a compact nonlinear reference.

Table 1: Trainable parameters and training protocols. Every residual also uses two fixed Xu–Randall constants; Q1 adds 128 non-trainable normalization states. Compact models are near-sized across families and exactly size-matched within each direct/residual pair.
<table><tr><td>Model family</td><td>Six inputs</td><td>Eight inputs</td><td>Training protocol</td></tr><tr><td>Xu-Randall</td><td>0</td><td>0</td><td>Fixed calibration</td></tr><tr><td>Linear residual</td><td>7</td><td>9</td><td>Adam, 30 epochs</td></tr><tr><td>Compact MLP, direct or residual</td><td>119</td><td>203</td><td>Adam, 30 epochs</td></tr><tr><td>XYZ QNN, direct or residual</td><td>109</td><td>201</td><td>Adam, 30 epochs</td></tr><tr><td>ZZXY QNN, direct or residual</td><td>114</td><td>200</td><td>Adam, 30 epochs</td></tr><tr><td>Q1, direct or residual</td><td>8,961</td><td>9,089</td><td>Adam, at most 300 epochs</td></tr></table>

The stronger reference adapts Grundner’s cell-based Q1 architecture [1]: $d  6 4  6 4  6 4  1$ with tanh, LeakyReLU of slope 0.2, and tanh hidden activations. Batch normalization follows the second activation. Every Dense kernel has L1 and L2 penalties of 0.004749 and 0.008732. The network has 128 non-trainable normalization states in addition to Table 1’s parameters. Direct Q1 uses seeded Glorot kernels and zero biases; residual Q1 also zeroes the final kernel. This adapts the published architecture to the present features, targets, and sample size.

Q1 uses a numerical target scale of 100 to retain the scale of its inherited regularization. Its direct output $g _ { \theta }$ minimizes $\mathrm { M S E } ( g _ { \theta } , 1 0 0 z ) + L _ { \mathrm { r e g } }$ and predicts $g _ { \theta } / 1 0 0$ . Its residual output supplies the dimensionless logit correction and minimizes $\mathrm { M S E } ( 1 0 0 \hat { z } , 1 0 0 z ) + L _ { \mathrm { r e g } }$ . These losses use the transformed target, not the original Q1 paper’s physical percentage target.

## 3 Experimental design

## 3.1 Data, input configurations, and matched replicates

We use the released cell-based ICON/DYAMOND-derived arrays accompanying Pastori et al. [10], without rerunning simulations or coarse-graining. Our loader retains rows with $q _ { l } + q _ { i } >$ $1 0 ^ { - 8 } \mathrm { k g } \mathrm { k g } ^ { - 1 }$ ; evaluation is conditional on this condensate-filtered distribution. The six inputs are specific humidity $q _ { v } ,$ liquid water $q _ { l } .$ , ice $q _ { i } .$ , temperature $T _ { a }$ , pressure $p ,$ and horizontal wind speed $h _ { w } = \sqrt { u ^ { 2 } + v ^ { 2 } }$ . Eight-input order is $( q _ { v } , q _ { l } , q _ { i } , T _ { a } , p , z _ { g } , h _ { w } , f )$ , adding geometric height and the Coriolis parameter. Compact models use the inherited fixed input transforms; Q1 uses raw selected variables standardized using training-set means and population standard deviations only. All models share $\tau$ (Appendix A).

Each of three replicates contains 10,000 training, 2,000 validation, and 20,000 test cells. Training and validation rows are disjoint selections from the released training shards; test rows come from testing shards. Sampling approximately balances 64 shards in each pool. Root seed 12345 generates separate data, initialization, and shufle seeds. Models share rows and batch permutations within a replicate; six- and eight-input settings use byte-identical split manifests, verified by SHA-256. These matched replicates vary both selected rows and initialization and are not temporal cross-validation folds. Test sets across replicates can overlap. The ten models in each setting yield 60 evaluation records, including six fixed-baseline evaluations. The Xu–Randal diagnostic continues to use its original five inputs in both settings.

## 3.2 Training budgets, initialization, and checkpoint selection

Compact trainable models use Adam [13], learning rate $1 0 ^ { - 3 }$ , epsilon $1 0 ^ { - 7 }$ , batch size 100, and 30 epochs. QNN angles are initialized uniformly on $[ - \pi / 2 , \pi / 2 ]$ and shared across a circuit’s direct/residual pair within each replicate. Direct heads are randomly initialized; residual heads start at zero. Compact MLPs retain the reference random initialization except for their residual final layer.

Table 2: Physical-space performance, mean ± sample SD over three matched replicates of 20,000 test cells each. All QNNs use local Z readout. Compact models share 30 epochs; $\mathrm { Q 1 }$ uses its separate longer protocol.
<table><tr><td rowspan="2">Model</td><td colspan="2">Six inputs</td><td colspan="2">Eight inputs</td></tr><tr><td>MSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>MSE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>Xu-Randall</td><td> $0 . 0 4 8 0 4 \pm 0 . 0 0 0 6 7$ </td><td> $0 . 5 0 4 \pm 0 . 0 0 8$ </td><td> $0 . 0 4 8 0 4 \pm 0 . 0 0 0 6 7$ </td><td> $0 . 5 0 4 \pm 0 . 0 0 8$ </td></tr><tr><td>Compact MLP, direct</td><td> $0 . 0 4 2 7 7 \pm 0 . 0 0 5 9 9$ </td><td> $0 . 5 5 8 \pm 0 . 0 6 2$ </td><td> $0 . 0 2 7 5 3 \pm 0 . 0 0 8 1 3$ </td><td> $0 . 7 1 5 \pm 0 . 0 8 5$ </td></tr><tr><td> $\mathrm { X Y Z ~ Q N N , ~ d i r e c t }$ </td><td> $0 . 0 2 5 4 7 \pm 0 . 0 0 5 3 0$ </td><td> $0 . 7 3 7 \pm 0 . 0 5 4$ </td><td> $0 . 0 2 1 6 5 \pm 0 . 0 0 3 7 4$ </td><td> $0 . 7 7 6 \pm 0 . 0 3 8$ </td></tr><tr><td> $\mathrm { Z Z X Y ~ Q N N , ~ d i r e c t }$ </td><td> $0 . 0 2 0 0 2 \pm 0 . 0 0 3 2 9$ </td><td> $0 . 7 9 3 \pm 0 . 0 3 4$ </td><td> $0 . 0 1 8 1 5 \pm 0 . 0 0 1 6 8$ </td><td> $0 . 8 1 2 \pm 0 . 0 1 7$ </td></tr><tr><td> $\mathrm { X R } + \mathrm { l i n e a r }$ </td><td> $0 . 0 1 6 6 3 \pm 0 . 0 0 0 0 8$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 0$ </td><td> $0 . 0 1 6 6 7 \pm 0 . 0 0 0 1 4$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { X R + c o m p a c t \ M L P }$ </td><td> $0 . 0 1 3 2 2 \pm 0 . 0 0 0 3 9$ </td><td> $0 . 8 6 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 3 1 0 \pm 0 . 0 0 0 1 7$ </td><td> $0 . 8 6 5 \pm 0 . 0 0 2$ </td></tr><tr><td> $\mathrm { X R + X Y Z ~ Q N N }$ </td><td> $0 . 0 1 1 2 4 \pm 0 . 0 0 0 1 7$ </td><td> $0 . 8 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 0 8 2 \pm 0 . 0 0 0 2 2$ </td><td> $0 . 8 8 8 \pm 0 . 0 0 2$ </td></tr><tr><td> $\mathrm { X R } + \mathrm { Z Z X Y } \mathrm { Q N N }$ </td><td> $0 . 0 1 2 0 3 \pm 0 . 0 0 0 1 2$ </td><td> $0 . 8 7 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 1 0 1 \pm 0 . 0 0 0 4 4$ </td><td> $0 . 8 8 6 \pm 0 . 0 0 5$ </td></tr><tr><td>Q1, direct</td><td> $0 . 0 1 0 0 3 \pm 0 . 0 0 0 5 5$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 9 4 1 \pm 0 . 0 0 0 4 1$ </td><td> $0 . 9 0 3 \pm 0 . 0 0 4$ </td></tr><tr><td> $\mathrm { X R } + \mathrm { Q 1 }$ </td><td> $0 . 0 0 7 7 8 \pm 0 . 0 0 0 0 6$ </td><td> $0 . 9 2 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 7 9 1 \pm 0 . 0 0 0 0 8$ </td><td> $0 . 9 1 8 \pm 0 . 0 0 1$ </td></tr></table>

Q1 uses Adam with learning rate $4 . 3 3 \times 1 0 ^ { - 4 }$ , epsilon 0.1, batch size 100, and at most 300 epochs. Its early-stopping counter resets after improvement exceeding $1 0 ^ { - 6 }$ in unregularized validation MSE, with patience 30. Independently of this tolerance, each trainer saves the earliest epoch attaining the exact minimum validation MSE. Only that checkpoint is tested. Direct transformed predictions are clipped to [0, 1] before inversion. The shared compact-model protocol and longer Q1 protocol answer diferent comparison questions; neither stopping rule certifies optimization convergence.

## 3.3 Evaluation metrics and paired statistics

Physical-space test MSE is the primary metric. We also report $\begin{array} { r } { R ^ { 2 } = 1 - \sum _ { i } ( \hat { c } _ { i } - c _ { i } ) ^ { 2 } / \sum _ { i } ( c _ { i } - } \end{array}$ $\bar { c } ) ^ { 2 } .$ , histogram Hellinger distance with 100 equal bins on [0, 1], and empirical one-dimensional Wasserstein distance. MSE and $R ^ { 2 }$ assess sample-wise errors and are reported in Table $2 ;$ the distributional distances compare marginal physical cloud-fraction distributions and are reported separately in Table 3. Lower values are better for both distances, whereas higher $R ^ { 2 }$ is better. On the same test set, MSE and $R ^ { 2 }$ give the same ranking.

We report actual replicate values, means, sample standard deviations, and win counts. Percentage improvements are calculated within replicate k as $1 0 0 ( \mathrm { M S E } _ { \mathrm { r e f } , k } - \mathrm { M S E } _ { \mathrm { c a n d } , k } ) / \mathrm { M S E } _ { \mathrm { r e f } , k }$ before averaging, rather than taking a ratio of mean MSEs. Three repeats support descriptive comparisons, not significance claims or uncertainty estimates over independent climate realizations. The manuscript data audit recomputes metrics from saved predictions and checks aligned targets, manifests, residual reconstruction, and selected validation minima.

## 4 Results

## 4.1 Residual prediction improves both compact QNN architectures

Residual prediction reduced test MSE in all twelve matched comparisons across the two circuits and input settings (Fig. 3a–d; Table 2). With six inputs, XYZ changed from mean MSE 0.02547 to 0.01124 and ZZXY from 0.02002 to 0.01203. The mean paired reductions were 54.3 ± 11.3% and $3 8 . 8 \pm 9 . 7 \%$ . With eight inputs, the corresponding reductions were $4 9 . 2 \pm 7 . 5 \%$ and $3 8 . 9 \pm 7 . 0 \%$ without adding trainable parameters within a circuit family.

Lower sample-wise MSE did not imply uniformly better distributional agreement (Table 3; lower is better for both distances). For eight-input XYZ, residual prediction increased mean Hellinger distance from 0.0713 to 0.0802, indicating worse agreement by this measure, but decreased mean Wasserstein distance from 0.0220 to 0.0197, indicating better agreement. The fixed Xu–Randall baseline also had lower mean Wasserstein distance than several learned models despite much larger MSE (Table 2). These distances compare the overall distributions without retaining the correspondence between individual predictions and targets. The residual benefit therefore concerns sample-wise prediction accuracy; the distributional diagnostics did not improve uniformly.

![](images/460db28deca98278aaa57bb4f5493be34b14c5194655ff756a23ac2fcc28ea7e.jpg)  
Figure 2: Test-set performance across input configurations. Each row groups six-input (open) and eightinput (filled) bars, whose lengths show mean physical-space MSE. Small points represent three matched replicates; whiskers show $\pm$ one sample SD $( n = 3 , 2 0 , 0 0 0$ test cells per replicate). QNNs are blue, Q1 adaptations ochre, and remaining controls gray. Every residual model is anchored to Xu–Randall; all QNNs use local Z readout. Q1 uses a larger architecture and a separate, longer training protocol. Axis values are expressed in units of $\mathrm { { 1 0 ^ { - 2 } } }$ ; lower values indicate lower error. Source data: runs.csv and summary.csv.

## 4.2 Classical residual controls also benefit, while QNNs outperform the selected compact MLP

The compact residual MLP also improved in every replicate (Fig. 2). Its mean MSE fell from 0.04277 to 0.01322 with six inputs and from 0.02753 to 0.01310 with eight. Even the sevenparameter linear correction attained MSE 0.01663; its nine-parameter version attained 0.01667. These results establish that useful corrections to the inherited baseline can be learned with classical models.

Within the shared 30-epoch protocol, residual QNNs nevertheless outperformed the selected residual MLP in every pair. XYZ reduced its MSE by 14.9 ± 2.5% with six inputs and $1 7 . 4 \pm 0 . 9 \%$ with eight; ZZXY reductions were $9 . 0 \pm 1 . 9 \%$ and $1 6 . 0 \pm 3 . 2 \%$ . Parameter counts were comparable but not identical across families (Table 1). The result favors these circuits over this compact classical control under the specified protocol, without establishing superiority over the class of similarly sized classical models.

Table 3: Distributional distances in physical cloud-fraction space. Values are means $\pm$ sample SD over three matched replicates, with 20,000 test cells per replicate. Both Hellinger and Wasserstein distances are lower-is-better (↓), unlike $R ^ { 2 }$ . Hellinger uses 100 equal bins on [0, 1]; Wasserstein is the empirical one-dimensional distance. Models and training protocols match Table 2. All QNNs use local Z readout.
<table><tr><td rowspan="3">Model</td><td colspan="2">Six inputs</td><td colspan="2">Eight inputs</td></tr><tr><td>Hellinger ↓</td><td> $\mathrm { W a s s e r s t e i n \downarrow }$ </td><td></td><td>Hellinger ↓ Wasserstein ↓</td></tr><tr><td>Xu-Randall</td><td> $0 . 0 9 7 2 \pm 0 . 0 0 1 1$ </td><td> $0 . 0 1 8 9 \pm 0 . 0 0 1 2$ </td><td> $0 . 0 9 7 2 \pm 0 . 0 0 1 1$ </td><td> $0 . 0 1 8 9 \pm 0 . 0 0 1 2$ </td></tr><tr><td>Compact MLP, direct</td><td> $0 . 2 3 2 0 \pm 0 . 0 6 5 5$ </td><td> $0 . 0 7 1 6 \pm 0 . 0 1 4 1$ </td><td> $0 . 2 1 3 0 \pm 0 . 0 9 0 2$ </td><td> $0 . 0 3 3 9 \pm 0 . 0 0 2 2$ </td></tr><tr><td>XYZ QNN, direct</td><td> $0 . 0 8 0 0 \pm 0 . 0 0 8 4$ </td><td> $0 . 0 2 4 7 \pm 0 . 0 0 1 8$ </td><td> $0 . 0 7 1 3 \pm 0 . 0 0 7 4$ </td><td> $0 . 0 2 2 0 \pm 0 . 0 0 1 9$ </td></tr><tr><td>ZZXY QNN, direct</td><td> $0 . 0 9 0 8 \pm 0 . 0 0 6 5$ </td><td> $0 . 0 2 7 9 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 7 7 4 \pm 0 . 0 0 9 6$ </td><td> $0 . 0 2 4 9 \pm 0 . 0 0 1 4$ </td></tr><tr><td> $\mathrm { X R } + \mathrm { l i n e a r }$ </td><td> $0 . 0 8 6 5 \pm 0 . 0 0 2 3$ </td><td> $0 . 0 2 4 6 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 8 4 5 \pm 0 . 0 0 2 6$ </td><td> $0 . 0 2 4 2 \pm 0 . 0 0 0 9$ </td></tr><tr><td> $\mathrm { X R + c o m p a c t \ M L P }$ </td><td> $0 . 0 9 9 3 \pm 0 . 0 0 4 5$ </td><td> $0 . 0 2 8 9 \pm 0 . 0 0 1 3$ </td><td> $0 . 0 8 5 7 \pm 0 . 0 1 0 3$ </td><td> $0 . 0 2 3 3 \pm 0 . 0 0 3 5$ </td></tr><tr><td> $\mathrm { X R + X Y Z ~ Q N N }$ </td><td> $0 . 0 8 3 1 \pm 0 . 0 0 6 1$ </td><td> $0 . 0 2 1 4 \pm 0 . 0 0 3 2$ </td><td> $0 . 0 8 0 2 \pm 0 . 0 0 5 9$ </td><td> $0 . 0 1 9 7 \pm 0 . 0 0 1 2$ </td></tr><tr><td> $\mathrm { X R + Z Z X Y ~ Q N N }$ </td><td> $0 . 0 8 6 0 \pm 0 . 0 0 1 9$ </td><td> $0 . 0 2 3 7 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 8 2 7 \pm 0 . 0 0 6 0$ </td><td> $0 . 0 2 2 5 \pm 0 . 0 0 2 3$ </td></tr><tr><td> $\mathrm { Q 1 , d i r e c t }$ </td><td> $0 . 1 6 2 4 \pm 0 . 0 1 1 5$ </td><td> $0 . 0 1 4 0 \pm 0 . 0 0 2 5$ </td><td> $0 . 1 1 9 9 \pm 0 . 0 1 4 6$ </td><td> $0 . 0 1 3 0 \pm 0 . 0 0 3 6$ </td></tr><tr><td> $\mathrm { X R } + \mathrm { Q 1 }$ </td><td> $0 . 0 8 1 9 \pm 0 . 0 0 4 5$ </td><td> $0 . 0 1 2 7 \pm 0 . 0 0 2 5$ </td><td> $0 . 0 7 8 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 1 1 2 \pm 0 . 0 0 1 4$ </td></tr></table>

## 4.3 The stronger Grundner Q1 reference remains more accurate

Both Q1 adaptations achieved lower MSE than residual XYZ in every matched replicate (Fig. 2). Direct Q1 attained means of $0 . 0 1 0 0 3 \pm 0 . 0 0 0 5 5$ and $0 . 0 0 9 4 1 \pm 0 . 0 0 0 4 1$ for six and eight inputs. Residual Q1 improved further to $0 . 0 0 7 7 8 \pm 0 . 0 0 0 0 6$ and $0 . 0 0 7 9 1 \pm 0 . 0 0 0 0 8$ , with mean $R ^ { 2 }$ of 0.920 and 0.918. Its paired MSE reductions relative to residual XYZ were $3 0 . 8 \pm 1 . 3 \%$ and $2 6 . 8 \pm 1 . 9 \%$

Residual Q1 also beat direct Q1 in every replicate. Its advantage reinforces the usefulness of residual prediction while bounding the quantum-model comparison. Q1 difers in capacity, preprocessing, regularization, optimizer settings, and training duration. Selected direct Q1 epochs ranged from 110–134 for six inputs and 139–300 for eight; one run selected the training cap. Residual Q1 selected epochs 127–219 and 167–210. These are empirical accuracy references under their protocols, not a classical performance ceiling or an isolated test of network width.

## 4.4 Benefits of the eight-input configuration depend on the model

On matched splits, eight-input residual XYZ reduced mean MSE from 0.01124 to 0.01082, a $3 . 8 \pm 1 . 7 \%$ mean paired improvement (Fig. 3e). Residual ZZXY improved from 0.01203 to 0.01101, corresponding to a mean paired reduction of $8 . 5 \pm 3 . 7 \%$ . Both circuits improved in all three replicates, whereas each direct QNN improved in only two despite lower mean MSE. Benefits were not uniform across families: linear residual MSE increased slightly, and residual Q1 was approximately 1.7% worse. Six inputs were preferred in all three residual Q1 pairs.

These diferences describe complete configurations. Eight-input QNNs change the variables, qubit count, and paper-specified depth together; the compact MLP shape also changes. Consequently, improvements cannot be assigned to geometric height or the latitude-related input alone. The unchanged Xu–Randall predictions confirm the aligned baseline, while the model-dependent outcomes delimit the benefit of the extended configuration.

Direct → residual prediction (6/8 inputs)  
![](images/d84a47ef4da9e9761396dc9e573a224807a7d59217b75e2f89c71c204776e1f4.jpg)

e Eight versus six inputs  
![](images/0fb4f99a6c6f494c78e86e4af37554fae16aeae114ff604cb324980574a9f68a.jpg)  
n = 3 matched replicates; 20,000 test samples each. Positive reductions indicate lower MSE.

Figure 3: Paired efects of residual prediction and input configuration. a–d, direct-to-residual QNN comparisons for six-input ${ \mathrm { X Y Z , } }$ six-input ZZXY, eight-input $\mathrm { { X Y Z , } }$ and eight-input ZZXY, respectively. Each thin line connects one matched replicate; small endpoints show its physical-space MSE. Large open circles (six inputs) and filled squares (eight inputs) show means, with whiskers indicating ± one sample SD. e, percentage MSE reductions for eight inputs relative to six, computed within replicate before averaging. Bars and right-hand labels show mean reductions; small points show individual pairs, and whiskers indicate ± one sample SD. Positive values favor eight inputs; colors identify QNNs (blue), Q1 (ochre), and other controls (gray). All panels use three matched replicates with 20,000 test cells each. QNN inputs, qubit count, and circuit depth change together in panel e.

## 5 Discussion

## 5.1 Interpreting physical residual prediction

The results support retaining a calibrated physical predictor within a compact QNN under the tested training budget. The benefit across classical residual controls places this finding within a broader strategy of correcting existing approximations. It extends the direct cloud-cover setting of Pastori et al. [10]; its usefulness does not depend on claiming that the correction strategy is exclusive to quantum circuits. Although residual QNNs outperform the selected near-sized MLP,

stronger Q1 references change the ranking.

This construction uses physical knowledge through a fixed predictor, rather than governingequation residuals in a physics-informed loss [14]. Bounded output also difers from enforcing analytic conservation constraints [15]. A learned correction can depart substantially from its anchor, and inherited calibration adds information beyond the 10,000 selected training examples. The experiment therefore establishes neither full physical consistency nor sample-eficiency scaling from an equal total-information budget.

The observed gain is attributable to the complete residual design. Relative to direct prediction, it changes the anchor, sigmoid output, and zero-head initialization simultaneously. The derivative $\partial \hat { z } / \partial r = \hat { z } ( 1 - \hat { z } )$ can attenuate gradients near the boundaries; zero head weights also make initial circuit-angle gradients zero, although head gradients can be nonzero. These properties suggest possible optimization efects but were not isolated experimentally. Holding the output map and initialization fixed while removing or replacing the physical anchor is the next discriminating test of its specific contribution.

## 5.2 Scope and next discriminating tests

The comparison covers one training size, three matched repeats, and particular architectures. Many compact-model validation histories continue improving near epoch 30. Q1’s longer protocol supplies an accuracy reference, but cannot isolate a capacity efect. A comparison with matched optimization budgets and stronger compact classical tuning would test how far the observed circuit-specific benefit persists [12].

The condensate filter and shard-balanced sampling further limit the evaluated distribution. Random cells from the released pools do not constitute new temporal or geographical blocking, and shared weather conditions can induce dependence across samples and repeats. Evaluating matched protocols on blocked periods or regions is needed to assess portability beyond the present split design. Input-number comparisons additionally remain confounded with depth and architecture changes.

Finally, all quantum predictions are exact state-vector expectations computed on classical accelerators. Parameter compactness does not measure simulation, encoding, measurement, or gradient costs. Finite shots, hardware noise, and hardware speed were not tested, and no online atmospheric coupling was performed. The supported outcome is an ofline predictive improvement under specified conditions; computational advantage and climate-model feedback require separate evidence.

## 6 Conclusion

A bounded logit correction around the fitted Xu–Randall anchor improved local-Z-readout XYZ and ZZXY QNNs in every matched replicate of the six- and eight-input experiments. These compact residual QNNs also surpassed the selected residual MLP under the shared 30-epoch protocol, while the larger Grundner-Q1 adaptations remained more accurate. The contribution is a reproducible prediction design and comparison that identifies both useful gains and the limits imposed by stronger classical references.

## A Exact implementation and supplementary results

## A.1 Transforms, physical baseline, and initialization

The inherited target transform and inverse are

$$
\begin{array} { l } { \displaystyle { \mathcal { T } ( c ) = \pi ^ { - 1 } \arcsin \left[ 2 \left( \frac { e ^ { B c ^ { A } } - 1 } { e ^ { B } - 1 } \right) ^ { C } - 1 \right] + \frac { 1 } { 2 } , } } \\ { \displaystyle { \mathcal { T } ^ { - 1 } ( z ) = \left[ \frac { \log \{ 1 + ( e ^ { B } - 1 ) u ( z ) ^ { 1 / C } \} } { B } \right] ^ { 1 / A } , \quad u ( z ) = \frac { 1 } { 2 } [ 1 + \sin \{ \pi ( z - \frac { 1 } { 2 } ) \} ] , } } \end{array}\tag{5}
$$

with $A = 1 . 2 9 4 0 7 9 1 3$ $B = - 3 . 2 0 0 1 1 0 1 5$ , and $C = 0 . 7 0 3 0 8 2 3 7$ . For humidity, liquid water, ice, and wind, define $L ( q ) = \log [ 1 + ( e - 1 ) ( q / q _ { \operatorname* { m a x } } ) ^ { \gamma } ]$ and input angle $\pi [ L ( q ) - L ( q _ { \mathrm { m i n } } ) ] / [ 1 -$ $L ( q _ { \mathrm { m i n } } ) ]$ . The triples $( \gamma , q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } )$ are $( 0 . 2 5 , 1 0 ^ { - 7 } , 0 . 0 2 5 )$ , (0.25, 0, 0.00145), (0.25, 0, 0.00055), and $( 0 . 5 , 0 . 0 0 1 5 , 1 1 5 )$ , respectively; moisture units are kg kg<sup>−1</sup> and wind units m s<sup>−1</sup>. Linear maps send temperature, pressure, and height ranges (180, 305) K, (4000, 105000) Pa, and (0, 20000) m to $[ 0 , \pi ]$ . Coriolis angle is arcsin $[ f / ( 2 \Omega ) ] + \pi / 2$ , with $\Omega = 7 . 2 9 2 1 \times 1 0 ^ { - 5 } \mathrm { s ^ { - 1 } }$ . Input angles outside fitted ranges are not clipped.

For Eq. (1), $\mathrm { R H } = \operatorname* { m a x } ( e / e _ { s } , 0 )$ and $e = p q _ { v } / ( 0 . 6 2 2 + 0 . 3 7 8 q _ { v } )$ . Saturation pressures are $e _ { w } =$ 611.21 e $\mathrm { x p } [ 1 7 . 2 6 9 ( T _ { a } - 2 7 3 . 1 5 ) / ( T _ { a } - 3 5 . 8 6 ) ]$ ] and $e _ { i } = 6 1 1 . 2 1 \exp [ 2 1 . 8 7 5 ( T _ { a } - 2 7 3 . 1 5 ) / ( T _ { a } - 7 . 6 6 ) ]$ Pa. Between 250.15 and 273.15 K, $e _ { s } = e _ { i } + ( e _ { w } - e _ { i } ) [ ( T _ { a } - 2 5 0 . 1 5 ) / 2 3 ] ^ { 2 }$ ; outside, $e _ { i }$ or $e _ { w }$ applies. Fixed transforms and calibration are inherited. Zero residual heads recover Eq. (2); clipping changed physical anchor values by less than $2 \times 1 0 ^ { - 5 }$ on the tests.

## References

[1] Arthur Grundner, Tom Beucler, Pierre Gentine, Fernando Iglesias-Suarez, Marco A. Giorgetta, and Veronika Eyring. Deep Learning Based Cloud Cover Parameterization for ICON. Journal of Advances in Modeling Earth Systems, 14(12):e2021MS002959, 2022. doi: 10.1029/2021ms002959. URL https://doi.org/10.1029/2021ms002959.

[2] Bjorn Stevens, Masaki Satoh, Ludovic Auger, Joachim Biercamp, Christopher S. Bretherton, Xi Chen, Peter Düben, Falko Judt, Marat Khairoutdinov, Daniel Klocke, Chihiro Kodama, Luis Kornblueh, Shian-Jiann Lin, Philipp Neumann, William M. Putman, Niklas Röber, Ryosuke Shibuya, Benoit Vanniere, Pier Luigi Vidale, Nils Wedi, and Linjiong Zhou. DYAMOND: the DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains. Progress in Earth and Planetary Science, 6 (1):61, 2019. doi: 10.1186/s40645-019-0304-z. URL https://doi.org/10.1186/s40645-019-0304-z.

[3] Günther Zängl, Daniel Reinert, Pilar Rípodas, and Michael Baldauf. The ICON (ICOsahedral Non-hydrostatic) modelling framework of DWD and MPI-M : Description of the non-hydrostatic dynamical core. Quarterly Journal of the Royal Meteorological Society, 141(687):563–579, 2015. doi: 10.1002/qj.2378. URL https://doi.org/10.1002/qj.2378.

[4] Kuan-Man Xu and David A. Randall. A Semiempirical Cloudiness Parameterization for Use in Climate Models. Journal of the Atmospheric Sciences, 53(21):3084–3102, 1996. doi: 10.1175/1520-0469(19 96)053<3084:ascpfu>2.0.co;2. URL https://doi.org/10.1175/1520-0469(1996)053<3084: ascpfu>2.0.co;2.

[5] Stephan Rasp, Michael S. Pritchard, and Pierre Gentine. Deep learning to represent subgrid processes in climate models. Proceedings of the National Academy of Sciences, 115(39):9684–9689, 2018. doi: 10.1073/pnas.1810286115. URL https://doi.org/10.1073/pnas.1810286115.

[6] Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and Prabhat. Deep learning and process understanding for data-driven Earth system science. Nature, 566(7743):195–204, 2019. doi: 10.1038/s41586- 019- 0912-1. URL https://doi.org/10.1038/s41586-019-0912-1.

[7] Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, and Vipin Kumar. Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems. ACM Computing Surveys, 55(4):1–37, 2022. doi: 10.1145/3514228. URL https://doi.org/10.1145/3514228.

[8] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. Data re-uploading for a universal quantum classifier. Quantum, 4:226, 2020. doi: 10.22331/q-2020-02-06-226. URL https://doi.org/10.22331/q-2020-02-06-226.

[9] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. Efect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 103(3):032430, 2021. doi: 10.1103/physreva.103.032430. URL https://doi.org/10.1103/physreva.103.032430.

[10] Lorenzo Pastori, Arthur Grundner, Veronika Eyring, and Mierk Schwabe. Quantum neural networks for cloud cover parameterizations in climate models. Machine Learning: Earth, 2(1):015008, 2026. doi: 10.1088/3049-4753/ae4981. URL https://doi.org/10.1088/3049-4753/ae4981.

[11] M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, and Patrick J. Coles. Variational quantum algorithms. Nature Reviews Physics, 3(9):625–644, 2021. doi: 10.1038/s42254-021-00348-9. URL https://doi.org/10.1038/s42254-021-00348-9.

[12] Joseph Bowles, Shahnawaz Ahmed, and Maria Schuld. Better than classical? The subtle art of benchmarking quantum machine learning models, 2024. URL https://arxiv.org/abs/2403.07059.

[13] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization, 2014. URL https://arxiv.org/abs/1412.6980.

[14] George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021. doi: 10.1038/s42254 -021-00314-5. URL https://doi.org/10.1038/s42254-021-00314-5.

[15] Tom Beucler, Michael Pritchard, Stephan Rasp, Jordan Ott, Pierre Baldi, and Pierre Gentine. Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems. Physical Review Letters, 126 (9):098302, 2021. doi: 10.1103/physrevlett.126.098302. URL https://doi.org/10.1103/physrevlet t.126.098302.