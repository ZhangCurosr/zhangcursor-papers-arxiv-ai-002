# Quantum Versus Classical Machine Learning for Neuropsychiatric Disorders: A Feature-Matched Multi-Cohort Evaluation

Utsav Poudel , Anuj Nepal , Vyas Viswanath, and Subramaniyaswamy Vairavasundaram

Abstract—Resting-state functional connectivity is widely used in machine learning-based classification of neuropsychiatric disorders. Quantum machine learning models have been proposed for functional connectivity data, but few studies have evaluated them under leave-site-out validation or against matched classical models. We compared six quantum model families with classical ones using identical correlation-derived features across four public cohorts (4,308 subjects). Leave-site-out validation was used for the Autism Brain Imaging Data Exchange (ABIDE I), ADHD-200 (attention-deficit/hyperactivity disorder), and REST-meta-MDD (major depressive disorder) cohorts, while the UCLA Consortium for Neuropsychiatric Phenomics (UCLA-CNP) was evaluated using stratified 10-fold cross-validation. The quantum kernel was statistically indistinguishable from a matched classical kernel across all four cohorts, with the average accuracy difference within 1.4 percentage points at 95% confidence. On ABIDE I, a hybrid quantum convolutional network achieved similar accuracy to its classical counterpart (0.645 vs. 0.641), while a trainable quantum kernel exceeded its classical counterpart by 0.064 in area under the curve (AUC). Quantum convolutional networks also achieved higher AUC than their matched classical networks on this cohort and showed smaller train–test gaps on all four cohorts, without an accuracy gain. Overall, most quantum models matched their classical counterparts, but none consistently outperformed them. This study, among the first quantum evaluations of ABIDE I under leave-site-out validation, provides a feature-matched benchmark of quantum and classical models comprising 284 matched comparisons. We further analyse fidelity-based graph construction and account for the observed quantum–classical parity through effective-dimension and eigenspectrum analyses. Code and per-fold results are available at https://github.com/r11up/qml-fmri-baseline.

Index Terms—Functional magnetic resonance imaging, graph neural networks, machine learning, mental disorders, quantum computing.

## I. INTRODUCTION

Artificial intelligence is increasingly being applied to psychiatric diagnosis, prognosis, and decision support systems [1], and resting-state functional magnetic resonance imaging (rsfMRI) is widely used as one of the few tools that enable noninvasive analysis of intrinsic functional brain activity. Most rs-fMRI pipelines begin by building a functional connectivity graph whose edges encode statistical dependency between regions of interest (ROIs). The Pearson correlation coefficient remains one of the most widely used measures because of its computational efficiency and interpretability [2]. However, it captures only linear, pairwise relationships and may overlook non-linear or higher-order dependencies between brain regions. For conditions with subtle, heterogeneous presentation, such as autism spectrum disorder (ASD) [3], non-linear dependence may carry complementary information, as reported for schizophrenia [4], but whether it improves generalisation to unseen acquisition sites remains an open question.

Graph neural networks (GNNs) address this limitation through a different mechanism. Rather than replacing the linear edge measure, they keep the correlation graph and recover non-linearity downstream, composing non-linear functions of a node’s neighbourhood across successive messagepassing layers. This paper investigates whether introducing non-linearity at the connectivity level, in the graph edges themselves, is advantageous compared with introducing it only within downstream models. The downstream route has a cost, since stacking layers runs into over-smoothing, where repeated aggregation over a dense, noisy graph pulls node representations toward a common mean [5]. The problem is worse when the topology is static and dense, which is the typical shape of a Pearson correlation matrix in fMRI.

A classical vector encoded into a quantum state maps into a Hilbert space where relations that were non-linear in the original coordinates can become close to linear [6]. We encode correlation-derived regional features as quantum states, use the fidelity between them to initialise the graph’s topology, and train a classical edge kernel and the classifier on that support. The design is therefore a quantum-fidelity-initialised adaptive graph rather than an end-to-end quantum one. The graph topology is determined by the encoding component of the circuit alone. Following standard connectivity pipelines, every model in this study receives features derived from the correlation matrix, so quantum and classical models see identical inputs and any difference between them reflects the similarity function rather than the input representation.

This design sits between two bodies of work that have, until recently, developed largely along separate tracks: classical adaptive graph learning, which treats the brain connectivity graph as a learnable object, and quantum machine learning applied to neuroimaging and other healthcare data, which has rarely been evaluated against matched classical baselines under leave-site-out validation. This work brings the two together and makes the following contributions:

• A matched, multi-cohort benchmark: a featurematched comparison of six quantum model families with classical models across four public cohorts, using leavesite-out validation where cohort structure permits, among the first such evaluations to our knowledge.

• An analysis of fidelity-based graph construction: a theoretical account of common-unitary cancellation and register-width-dependent concentration.

• A spectral comparison of quantum and classical kernels: an effective-dimension, eigenspectrum and predictive comparison of quantum-fidelity and radial basis function (RBF) kernels under identical features.

• A systematic evaluation of quantum–classical differences: 284 matched comparisons across cohorts and metrics, with prediction-degeneracy screening and multipletesting correction, supported by released code and perfold results.

## II. RELATED WORK

Machine learning has advanced quickly across psychiatry over the past decade [7], and two lines of that work converge on this problem: classical adaptive graph learning, which treats the connectivity graph as a learnable object rather than a fixed correlation matrix, and quantum machine learning applied to neuroimaging and other biomedical data. Neither has yet produced a system that builds a functional-connectivity graph from quantum measurements and trains it against a clinical classification objective.

A. Classical Adaptive Graph Learning for Functional Connectivity

The limits of static correlation graphs have pushed several groups toward making the graph itself a learnable parameter, optimised jointly with the network weights. AGMGC [8] replaces a predefined connectivity assumption with a learner module that infers structure from raw time series, augments it with a partial-correlation graph, and reaches 84.65% accuracy on the Autism Brain Imaging Data Exchange (ABIDE), ahead of population-graph baselines. Its adjacency is still a softmax over classically learned projections: a more flexible classical similarity function, but a classical one.

A second trend folds in auxiliary modalities to refine the graph. DAGMNet [9] fuses structural MRI (sMRI), fMRI, and phenotypic data (age, sex, and similar covariates) through a phenotype-guided population-graph module that prunes connections unsupported by clinical priors, reaching 91.59% accuracy and a 96.80% area under the receiver operating characteristic curve (AUC) on ABIDE I. Section X-F examines what that figure measures. Auxiliary context has also been paired with EEG, where a dual-tower model combining EEG with a graph-encoded environmental prior improved affectivestate classification over EEG alone [10].

B. Quantum Machine Learning in Neuroimaging and Healthcare

A 2025 systematic review of quantum deep learning in neuroinformatics describes itself as a pioneering effort to compare quantum and classical approaches on brain data [11], and a 2025 survey of quantum machine learning pairs the promise of hybrid models with the noise and qubit-scaling limits of current hardware [12]. What does exist clusters around a few entry points into brain data.

One entry point is the direct application of a quantum or hybrid circuit to a time series or an image. Choi et al. use a onedimensional convolutional neural network (1-D CNN) front end with a quantum convolutional neural network (QCNN) on rs-fMRI [13]; Park et al. use a quantum time-series transformer on the same modality [14]. For ASD, Saranya and Menaka [15], Bagoun et al. [16] and Turjya and Fawakherji [17] apply quantum classifiers to electroencephalography (EEG). None builds an inter-regional adjacency matrix.

The one quantum model we found evaluated on ABIDE I functional connectivity is QG-WRN [18], which places a four-qubit parameterised circuit inside a graph convolutional residual network on the 116-region Automated Anatomical Labelling (AAL-116) atlas and reports 68.49% accuracy on a random 80/20 split of 775 subjects, a protocol its authors note does not decouple inter-site scanner variance.

## C. Quantum Architectures Beyond Static Fidelity

The constraint this paper measures, that pairwise state fidelity concentrates as the register widens, is specific to kernels built from static pairwise overlaps, and a recent line of work avoids it by changing the quantum primitive rather than the register. The quantum time-series transformer of Park et al. [14] realises attention through a Linear Combination of Unitaries and the Quantum Singular Value Transformation. Applied to resting-state fMRI, it processes temporal dependencies directly instead of collapsing them into a similarity matrix, operates at polylogarithmic complexity, and reports performance competitive with classical transformers on the Adolescent Brain Cognitive Development (ABCD) and UK Biobank cohorts.

It targets the bottleneck that Section VIII-A measures without widening the register, and reports its gains as most pronounced in small samples, the regime in which our quantum arm overfits less (Section X-A) without gaining accuracy (Section X-D). We do not evaluate it here and identify it as the most natural next comparison for this benchmark.

## D. Positioning of This Work

Against the classical adaptive graph learning of Section II-A, the difference is the similarity function: quantum fidelity in place of softmax attention or phenotype gating. Against the quantum-neuroimaging classifiers of Section II-B, the difference is architectural: an explicit adjacency, fidelityinitialised and classically weighted, feeding a graph-attention network rather than a per-region classifier. Against QG-WRN [18], the closest prior quantum model on the same cohort, the difference is protocol: whole acquisition sites are held out, the

200-region Craddock (CC200) parcellation is used, and every quantum model is set against a classical model matched on features, capacity, and training budget.

## E. Network-Level Findings Reported for These Cohorts

Table I lists the functional networks the clinical literature reports as disrupted in each disorder, and Fig. 1 renders them on cortical and subcortical surfaces, each Yeo network [19] coloured by its rank in that literature; the panel for the transdiagnostic UCLA Consortium for Neuropsychiatric Phenomics (UCLA-CNP) instead shows the CC200/Yeo parcellation, which doubles as the atlas reference throughout. For depression, the direction of default-mode change is contested, with the largest multi-site analysis reporting reduced connectivity in recurrent depression [20]. These are prior findings that situate the cohorts, not outputs of any model here; Section XI explains why models near chance cannot support networklevel claims.

## III. THEORETICAL FRAMEWORK

## A. Graph Representation of Functional Connectivity

The human brain is modelled as a graph $G = ( V , E )$ whose nodes $V = \{ v _ { 1 } , \ldots , v _ { N } \}$ are the N brain regions defined by a brain parcellation, each carrying a feature vector $x _ { i } \in \mathbb { R } ^ { \bar { T } }$ the blood-oxygen-level-dependent (BOLD) signal time series over T timepoints. The edges E are the connections between regions, and the $N \times N$ adjacency matrix A gives in $A _ { i j }$ the strength of the functional connection between node i and node $j .$

In classical functional connectivity analysis, A is typically the Pearson correlation matrix:

$$
A _ { i j } ^ { \mathrm { c l a s s i c a l } } = \frac { \sum _ { t = 1 } ^ { T } ( x _ { i , t } - \bar { x } _ { i } ) ( x _ { j , t } - \bar { x } _ { j } ) } { \sqrt { \sum _ { t = 1 } ^ { T } ( x _ { i , t } - \bar { x } _ { i } ) ^ { 2 } } \sqrt { \sum _ { t = 1 } ^ { T } ( x _ { j , t } - \bar { x } _ { j } ) ^ { 2 } } }\tag{1}
$$

Here $\begin{array} { r } { \bar { x } _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } x _ { i , t } } \end{array}$ is the temporal mean of the signal of region i, so that $\bar { A _ { i j } ^ { \mathrm { c l a s s i c a l } } } \in [ - 1 , 1 ]$

Pearson correlation primarily captures linear, zero-lag, pairwise dependence, so it may not fully represent non-linear relationships, such as $x _ { j } ( t ) = x _ { i } ( t ) ^ { 2 }$ , or higher-order relationships involving more than two regions at once.

## B. Quantum Feature Maps and Hilbert Spaces

To compare regions through a non-linear similarity, we map each classical feature vector $x _ { i } = [ x _ { i , 1 } , \dotsc , x _ { i , d } ]$ into a quantum state $| \psi _ { i } \rangle$ in a Hilbert space H with a parameterised quantum circuit on d qubits [21]. We use angle encoding, which needs one qubit and one rotation per feature, so the register grows linearly with feature dimension, and which avoids the normalisation and deep state-preparation circuits that amplitude encoding requires. Each feature $x _ { i , k }$ becomes the angle of a rotation about the Y-axis on one qubit:

$$
\begin{array} { r } { | \psi ( x _ { i } ) \rangle = \bigotimes _ { k = 1 } ^ { d } R _ { y } ( x _ { i , k } ) | 0 \rangle , \quad R _ { y } ( \theta ) = \left[ \begin{array} { c c } { \cos \frac { \theta } { 2 } } & { - \sin \frac { \theta } { 2 } } \\ { \sin \frac { \theta } { 2 } } & { \cos \frac { \theta } { 2 } } \end{array} \right] . } \end{array}\tag{2}
$$

The amplitudes of the resulting state are products of $\cos ( x _ { i , k } / 2 )$ and sin $\iota ( x _ { i , k } / 2 )$ across qubits, so the state is a nonlinear function $\phi ( x )$ of the input features. Because a rotation is periodic in its angle, the kernel and variational models minmax scale each feature on the training sites before encoding, to $[ 0 , \pi ]$ and $[ 0 , \pi / 2 ]$ respectively, so that distinct feature values map to distinct states. To capture dependencies between features, we add controlled-NOT (CNOT) gates between qubits,

$$
\mathrm { C N O T } | q _ { 1 } q _ { 2 } \rangle = | q _ { 1 } , q _ { 1 } \oplus q _ { 2 } \rangle ,\tag{3}
$$

where $q _ { 1 }$ is the control qubit, $q _ { 2 }$ the target qubit, and ⊕ denotes addition modulo two; the gate ties the qubits into a joint state that cannot be factored into independent single-qubit descriptions.

## C. Quantum Fidelity as a Connectivity Metric

The design evaluated in this paper defines graph connectivity by the $\hbar d e l i t y$ between quantum-encoded regional representations, which are themselves derived from correlationbased features. The fidelity between two pure quantum states $| \psi _ { i } \rangle$ and $| \psi _ { j } \rangle$ , representing two brain regions, is the squared overlap:

$$
A _ { i j } ^ { \mathrm { q u a n t u m } } = \vert \langle \psi ( x _ { i } ) \vert \psi ( x _ { j } ) \rangle \vert ^ { 2 }\tag{4}
$$

Three properties of this metric matter for graph construction. Fidelity is equivalent to a quantum kernel $K ( x _ { i } , x _ { j } )$ , and because the feature map involves non-linear trigonometric terms, it defines a non-linear similarity function, whose relationship to linear and RBF kernels we assess empirically through matched performance, effective dimension, and eigenspectrum. Fidelity also measures the geometric similarity of the encoded states, a non-linear function of all of a region’s features at once. Finally, in high-dimensional Hilbert spaces, most pairs of states are close to orthogonal, a consequence of measure concentration, so fidelity can give a sparser adjacency than a dense correlation matrix. Sparsity does not establish denoising, however: concentration pushes every overlap toward zero whether or not the pair is related, so it can suppress informative relationships along with noise.

## D. Adaptive Graph Attention Mechanism

The fidelity matrix $A ^ { \mathrm { q u a n t u m } }$ supplies only a candidate topology: it is computed without reference to diagnostic labels, so it need not reflect the structure relevant to classification. We therefore use it to select candidate edges and let a graph attention network (GAT) [22] learn how strongly each contributes: in the principal experiments, the fidelity graph defines a fixed support that remains unchanged during training, while the edge weights and graph-classification parameters are optimised.

A standard GAT computes attention coefficients from node features alone,

$$
\alpha _ { i j } = { \mathrm { s o f t m a x } } _ { j } \left( \operatorname { L e a k y R e L U } ( a ^ { \top } [ \Theta h _ { i } \parallel \Theta h _ { j } ] ) \right) ,\tag{5}
$$

where $h _ { i }$ is the feature vector of node i, Θ a shared learnable weight matrix, a a learnable attention vector and ∥ denotes

Table I Table 1

concatenation. We modify this mechanism so that attention is restricted to the fidelity-selected candidate edges and combines three terms. Let $z _ { i }$ be the latent vector of region i, given by the per-qubit Pauli-Z expectation values of its encoded state; $\mathrm { S i m } ( z _ { i } , z _ { j } )$ the similarity of the latents of regions i and $j ;$ Learna $\mathrm { b l e } ( z _ { i } , z _ { j } )$ a learned function of the pair; $\mathrm { A t t n } _ { \mathrm { G A T } } ( z _ { i } , z _ { j } )$ the standard attention score above; $\sigma ( \cdot )$ a pointwise non-linearity; and $\alpha , ~ \beta$ and $\gamma$ learnable mixing weights. The hybrid attention coefficient is

$$
\begin{array} { r l } & { \alpha _ { i j } ^ { \mathrm { h y b r i d } } = \mathrm { s o f t m a x } _ { j } \Big ( \sigma \big ( \alpha \cdot \mathrm { S i m } ( z _ { i } , z _ { j } ) } \\ & { ~ + \beta \cdot \mathrm { L e a r n a b l e } ( z _ { i } , z _ { j } ) } \\ & { ~ + \gamma \cdot \mathrm { A t t n } _ { \mathrm { G A T } } ( z _ { i } , z _ { j } ) \big ) \Big ) } \end{array}\tag{6}
$$

The mixing weights $\alpha , \beta$ and $\gamma$ are distinct from the doubleindexed attention coefficients $\alpha _ { i j }$ and from the RBF kernel width $\gamma$ used for the classical kernels. In training, Sim is the cosine similarity of the latents and Learnable is a small feedforward network on the concatenated pair; their weighted sum, passed through a rectified linear unit (ReLU), forms the edge weight, which enters the GAT attention alongside the standard score.

## IV. METHODOLOGY: THE ARCHITECTURE UNDER TEST

## A. Dataset and Preprocessing

The architecture is developed on the Autism Brain Imaging Data Exchange, ABIDE I [23], [24], which contributes 1,035 subjects (505 ASD, 530 controls) across 20 acquisition sites after quality control. It is then evaluated on three further cohorts in Section VII: ADHD-200 [25] for attentiondeficit/hyperactivity disorder (ADHD), REST-meta-MDD [20] for major depressive disorder (MDD), and UCLA-CNP [26], for 4,308 subjects in total (Table I). The cohorts are reported separately and never pooled: their releases disagree on parcellation width (200 CC200 regions for ABIDE and REST-meta-MDD against 190 for the others) and they were preprocessed with different pipelines (the Configurable Pipeline for the Analysis of Connectomes (C-PAC), Athena, the Data Processing & Analysis for Brain Imaging (DPABI) toolbox, and our own), so pooling would confound preprocessing with site. The heterogeneity across scanner types, acquisition protocols, and demographics within each cohort is what the leave-siteout protocol of Section IV-D is designed to stress-test. Fold counts reflect sites carrying both classes, and Table I reports the subjects contained in these folds. This affects ADHD-200 alone, whose release holds 767 subjects, of which 619 fall in sites carrying both classes. The other 148 come from two control-only sites, which cannot serve as test folds because sensitivity, specificity, and AUC are undefined on a single class; they are used for training in every fold and never tested, so no test label is exposed and every quantum and classical model trains on the same subjects. The remaining three cohorts are used in full. UCLA-CNP has only two scanners, and a paired test on two folds cannot reach significance at any effect size, so it is evaluated with stratified 10-fold cross-validation; its folds are therefore within-site and its accuracies are not on the same scale as those of the other three. These four were selected because each releases per-subject diagnostic labels together with scanner or site identifiers, which the matched leave-site-out design requires; they are not a census of available rs-fMRI data.

COHORTS EVALUATED, DIAGNOSTIC GROUPS, AND BRAIN NETWORKS REPORTED AS DISRUPTED.
<table><tr><td rowspan=1 colspan=1>Cohort</td><td rowspan=1 colspan=1>Diagnosticgroups</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Folds</td><td rowspan=1 colspan=1>ROIs</td><td rowspan=1 colspan=1>Networks reporteddisrupted</td></tr><tr><td rowspan=1 colspan=1>ABIDE I</td><td rowspan=1 colspan=1>ASD</td><td rowspan=1 colspan=1>1,035</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>DMN ↓, frontoparietal↓, limbic/amygdala ↓</td></tr><tr><td rowspan=1 colspan=1>ADHD-200</td><td rowspan=1 colspan=1>ADHD</td><td rowspan=1 colspan=1>619</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>190</td><td rowspan=1 colspan=1>frontoparietal ↓, DMN(delayed suppression),subcortical/striatal ↓</td></tr><tr><td rowspan=1 colspan=1>REST-meta-MDD</td><td rowspan=1 colspan=1>MDD</td><td rowspan=1 colspan=1>2,428</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>DMN (directioncontested), salience ↑,frontoparietal ↓</td></tr><tr><td rowspan=1 colspan=1>UCLA-CNP</td><td rowspan=1 colspan=1>SCZ, BD,ADHD,HC</td><td rowspan=1 colspan=1>226</td><td rowspan=1 colspan=1>10*</td><td rowspan=1 colspan=1>190</td><td rowspan=1 colspan=1>frontoparietal↓↓,thalamocortical, auditory↑</td></tr></table>

<sup>∗</sup>Stratified 10-fold; all other cohorts are leave-site-out. For UCLA-CNP, all patients (124) are classified against HC (102). DMN: default-mode network; SCZ: schizophrenia; BD: bipolar disorder; HC: healthy controls.

For ABIDE I we use the version preprocessed with C-PAC: slice-timing and motion correction, nuisance regression over white matter, cerebrospinal fluid (CSF) and 24 motion parameters, bandpass filtering (0.01–0.1 Hz), and parcellation into 200 CC200 regions covering the cerebrum and cerebellum. All four cohorts are represented on the CC200 functional parcellation [27], whose regions are data-driven functional clusters rather than named anatomical structures. The atlas specifies 200 clusters, but the spatially constrained spectral clustering resolves to 190 non-empty parcels, which is why releases disagree on width.

Output: for each subject, a feature matrix $X _ { \mathrm { r a w } } \in \mathbb { R } ^ { 2 0 0 \times T }$ where $T$ ranges from 150 to 300 depending on site.

Dimensionality reduction: current noisy intermediate-scale quantum (NISQ) devices and simulators have limited qubit capacity for efficient gradient computation, and directly encoding $T$ ≈ 200 timepoints would need prohibitively deep circuits. We therefore reduce each region’s description to $n _ { \mathrm { q u b i t s } }$ components, eight in the reported configuration. In every variant, the matrix presented to principal component analysis (PCA) has the 200 regions as its observations, so the fit is per subject or per fold and never per region.

Two descriptions of a region and two fitting scopes are used. The first fits PCA on $X _ { \mathrm { r a w } } ~ \in ~ \mathbb { R } ^ { 2 0 0 \times T }$ , one fit per subject, with each region’s time course standardised beforehand so that components follow temporal shape rather than BOLD amplitude; it retains 59.9% of the variance. The second describes region i by its connectivity profile, row i of the subject’s 200 × 200 correlation matrix, and fits one basis on the stacked profiles of every training-site subject, a $\left( 2 0 0 N _ { \mathrm { t r a i n } } \right) \times 2 0 0$ matrix with $N _ { \mathrm { t r a i n } }$ the number of training subjects, applying it unchanged to the held-out site; it retains 91.1%. Fitting inside the fold preserves the site-held-out guarantee. Measured with no graph involved, these two choices, what a node describes and where its basis is fitted, move held-out AUC from 0.459 to 0.739, further than any quantum–classical difference measured in this study. The quantum model suite of Section VI does not use this compression; it consumes the supervised connectivity features described below.

![](images/e8ecf2048abab1af1f78f79e31c5a33b2cdfe88b827c838009bb999b53649c0f.jpg)  
Figure 1. Functional networks reported as disrupted in (a) ABIDE I, (b) ADHD-200, (c) REST-meta-MDD and (d) UCLA-CNP.

Final input: a node feature matrix $X \in \mathbb { R } ^ { 2 0 0 \times d }$ with $d =$ n<sub>qubits</sub>, so each brain region is described by d latent features, with $d = 8$ in the reported configuration and $d = 1 6$ in the width study of Section VIII-B.

Supervised feature selection: Unsupervised PCA maximises variance rather than class separation, so it can discard precisely the directions that distinguish groups, and published pipelines on these cohorts generally use supervised selection instead. Every matched quantum–classical comparison therefore uses the top-k connections ranked by absolute t-statistic (k = 8 for the matched models, $k = 2 { , } 0 0 0$ for the unrestricted ones), fitted on the training sites only. Each matched model thus receives Pearson correlation coefficients rather than BOLD time series, so quantum and classical models see identical inputs and differ only in the similarity they apply. Selection is performed inside each fold because ranking features on the full cohort would leak held-out labels into the feature stage, a form of leakage shown to inflate connectome-based prediction performance [28].

## B. System Architecture Overview

The evaluated system is a hybrid pipeline: a simulated quantum circuit handles state preparation and fidelity computation, and a classical processor handles edge learning and graph classification, as shown in Fig. 2. A quantum feature map generates node representations and graph support, which are subsequently processed by a classical GAT.

The Quantum Feature Encoder maps regional connectivity features into quantum states in a Hilbert space. Circuits are simulated with a batched $\mathrm { P y }$ Torch statevector simulator, with PennyLane as the reference implementation (Section IV-C). The register carries n qubits per region, one per retained PCA component; the value of n is determined through fidelitydensity and computational-cost analyses. Fidelity at $n = 1 6$ leaves too few usable pairs at three of the four floors tested (Section VIII-A), while wider variational circuits are affordable only up to a point (Section VIII-B). Every comparison in this paper uses $n = 8 ,$ which provides sufficient connectivity density while remaining computationally practical.

Fig. 3 shows the per-region circuit gate by gate: a layer of $R _ { y }$ rotations whose angles carry the input, a ring of CNOTs closing from $q _ { n - 1 }$ back to $q _ { 0 } ,$ , a trainable $R _ { z } R _ { x }$ layer, and a per-qubit Pauli-Z read-out. The encoding layer prepares

$$
| \psi ( x _ { i } ) \rangle = \bigotimes _ { k = 1 } ^ { n } R _ { y } ( \theta _ { k } \cdot x _ { i , k } ) | 0 \rangle\tag{7}
$$

where $\theta _ { k }$ is the trainable gain on the k-th encoding rotation. In the principal experiments, all gains remained fixed at their initial value of one. After the CNOT ring and the variational $R _ { z } R _ { x }$ layer (angles ϕ<sub>k</sub>), a per-qubit Pauli-Z measurement returns a latent vector $z _ { i } \in \mathbb { R } ^ { n }$ with entries $z _ { i , k } = \langle \psi _ { i } | Z _ { k } | \psi _ { i } \rangle$ where |ψ ⟩ is the final circuit state; this vector forms the node features, with the full state vector retained separately for the fidelity of Eq. (4). In that fidelity, every data-independent unitary applied after the encoding, the CNOT ring as well as the variational layer, cancels from the overlap, because it acts identically on both states (Section VIII-C). The encoder carries no decoder and no reconstruction objective; in the principal graph runs, it is kept at its initial parameters (Section IV-C).

The Adaptive Edge Learning Module learns edge weights over a graph support that fidelity establishes once at initialisation, recomputing the weights at every training step.

• Input: the collection of latent quantum vectors $\{ z _ { 1 } , \dots , z _ { 2 0 0 } \}$

• Mechanism: quantum fidelity (Eq. (4)) selects the candidate edges: the squared overlap of two regions’ states ranks how strongly they are connected, and the strongest pairs form the support.

During training, the edge weights on this support are computed from the latent representations rather than recalculated from state fidelity:

$$
W _ { i j } = \operatorname { R e L U } \left( \alpha \cdot \operatorname { C o s S i m } ( z _ { i } , z _ { j } ) + \beta \cdot \mathbf { M L P } ( z _ { i } \| z _ { j } ) \right)\tag{8}
$$

Here α and $\beta$ are learnable scalars, CosSim $( u , v ) \ =$ $u ^ { \top } v / ( \| u \| \| v \| )$ is cosine similarity, MLP is a learned multilayer perceptron (MLP) and ∥ denotes concatenation.

![](images/2c3b9c22294e8607c3efc24544cf41204199152802475a8f57f103741be762d0.jpg)  
Figure 2. The fidelity-graph architecture evaluated here, showing quantum feature encoding and adaptive graph construction.

• Sparsification: to control noise and keep the GAT’s computation manageable, we apply k-nearest-neighbour masking, keeping only the top $k = 2 0$ connections per node, 10% of the 200 regions, which keeps the graph sparse while leaving each node a choice of neighbours for attention to weight. The result is left directed and is not symmetrised: region i may retain an edge to j without the converse, since j may have twenty stronger neighbours of its own. Attention is computed over incoming edges, so the asymmetry is well defined.

The Graph Attention Network (GAT) performs the binary classification. Each node’s quantum latent becomes its initial GAT feature, $h _ { i } ^ { ( 0 ) } = z _ { i }$ , so gradients from the classification loss reach the classical attention layers and the adaptive-edge kernel. The resulting adjacency, A<sup>quantum</sup> in its support and W of Eq. (8) in its values, is passed to the GAT for propagation:

$$
h _ { i } ^ { \prime } = \alpha _ { i , i } \Theta h _ { i } + \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i , j } \Theta h _ { j }\tag{9}
$$

where $h _ { i } ^ { \prime }$ is the updated feature of node $i , \ \mathcal { N } ( i )$ its set of incoming neighbours in the sparsified graph and $\alpha _ { i , j }$ the attention coefficients, modulated by the edge weights of Eq. (8) on the fidelity-selected support.

The classifier is a two-layer GAT. The first layer maps each region’s n = 8-dimensional latent to 32 features in each of four attention heads, concatenated to 128; the second maps these to 32 features with a single head. Both use exponential linear unit (ELU) activations and dropout $p = 0 . 6$ , chosen to limit overfitting given the sample size. Global mean pooling gives a graph-level embedding, which a linear layer maps to the two classes with a softmax read-out.

## C. Training Setup

Only the classical components are trained. Before training, the PCA basis is fitted on the training sites, the circuit is evaluated once for every region, and the latents and the fidelityselected support are cached; each training step then passes the cached latents through edge learning and the GAT to the crossentropy loss $\begin{array} { r } { \mathcal { L } = - \overline { { \sum } } _ { c = 1 } ^ { 2 } \mathcal { \bar { \lfloor } } } \end{array}$ y<sub>c</sub> log ˆy<sub>c</sub>, where $y _ { c }$ is the one-hot label and $\hat { y } _ { c }$ the predicted class probability. Gradients from PyTorch autograd update the GAT, the edge-learning MLP, and $\alpha , \beta$ with Adam [29] (learning rate 0.01, weight decay $5 \times 1 0 ^ { - 4 }$ , 30 epochs). In the principal graph runs (all rows of Table II but the trained-encoder variant), the quantum encoder, with every encoding gain equal to one, and the fidelityderived topology therefore remain fixed throughout training, p̂while only the classical edge-learning and GAT parameters are updated; the circuits of the model suite (Section VI) are trained separately. For the kernel models, bandwidth is selected p̂<sub>by kernel-target alignment (KTA) [30] on the training sites of</sub> each fold, with no inner cross-validation loop, and reported metrics are averaged across folds. Algorithm 1 summarises one training step.

Implementation: Experiments used Python 3.13, with Py-Torch [31] for a batched statevector simulator, PyTorch Geometric for graph operations, and PennyLane as the reference quantum backend. Because the topology touches all 19,900 region pairs per subject, the simulator evolves every region of a subject as one batched tensor, and circuits that are trained take gradients through autograd; PennyLane, which executes region by region with adjoint differentiation, agrees with it to simulator precision but is too slow for the fold counts reported here. All experiments (four cohorts, six architectural families, 62 folds) ran on one laptop (Apple M4 Pro, 14 cores, 24 GB memory, CPU only); no circuit exceeded sixteen qubits, and the largest state vector held in memory was 0.16 GB, so no specialised hardware is needed to reproduce the results.

cancels in the fidelity   
|0⟩<sub>q0</sub> R<sub>y</sub>(θ<sub>0</sub>x<sub>0</sub>) ① R<sub>z</sub>R<sub>x</sub>(φ<sub>0</sub>) ⟨Z<sub>0</sub>⟩   
|0⟩<sub>q1</sub> R<sub>y</sub>(θ<sub>1</sub>x<sub>1</sub>) ① R<sub>z</sub>R<sub>x</sub>(φ<sub>1</sub>) ⟨Z<sub>1</sub>⟩   
|0⟩<sub>q2</sub> R<sub>y</sub>(θ<sub>2</sub>x<sub>2</sub>) ① R<sub>z</sub>R<sub>x</sub>(φ<sub>2</sub>) ⟨Z<sub>2</sub>⟩ z ∈ R<sup>n</sup>   
|0⟩<sub>q3</sub> R<sub>y</sub>(θ<sub>3</sub>x<sub>3</sub>) ① R<sub>z</sub>R<sub>x</sub>(φ<sub>3</sub>) ⟨Z<sub>3</sub>⟩   
|0⟩<sub>qn−1</sub> R<sub>y</sub>(θx) ① R<sub>z</sub>R<sub>x</sub>(φ) ⟨Z⟩   
|ψ(x)⟩   
1. encode 2. entangle 3. trainable 4. read-out

Algorithm 1 One hybrid training step on a fixed support   
Require: BOLD series $X _ { \mathrm { r a w } } \in \mathbb { R } ^ { 2 0 0 \times T }$ and label y of one subject;   
basis P fitted on the training sites of the fold; encoding gains   
$\{ \theta _ { k } \}$ , variational angles $\{ \phi _ { k } \} ,$ edge weights $( \alpha , \beta )$ , GAT weights   
Θ   
Ensure: parameters after one gradient step   
1: $X  \overset { \cdot } { P } \big ( \mathrm { p r o f i l e s } ( X _ { \mathrm { r a w } } ) \big )$ $\{ n = 8$ components}   
2: for each region $i = 1 , \dots ,$ 200 do {once, then cached}   
3: prepare $\vert \psi ( x _ { i } ) \rangle \colon R _ { y } ( \theta _ { k } x _ { i , k } )$ , CNOT ring, $R _ { z } R _ { x }$ layer   
4: $\smash { z _ { i } \gets \bigl ( \langle \dot { \psi } ( x _ { i } ) | \mathcal { Z } _ { k } | \dot { \psi } ( x _ { i } ) \rangle \bigr ) _ { k = 1 } ^ { n } }$   
5: end for   
6: if the topology is not yet initialised then {once, without gradi  
ent}   
7: $\dot { F } _ { i j } \gets | \langle \psi ( x _ { i } ) | \psi ( x _ { j } ) \rangle | ^ { 2 }$ for all pairs {Eq. (4)}   
8: keep the $k = 2 0$ highest-fidelity neighbours of each region   
{discrete top-k}   
9: end if   
10: for each retained edge $( i , j )$ do   
11: $W _ { i j }  \mathrm { R e L U } \big ( \alpha \mathrm { \ ` C o s S i m } ( z _ { i } , z _ { j } ) + \beta \mathrm { M L P } ( z _ { i } \| z _ { j } ) \big )$   
12: end for   
13: $\hat { y } \gets \mathrm { G A T } _ { \Theta } ( \{ z _ { i } \} , W )$ with global mean pooling   
14: $\mathcal { L }  \mathcal { C }$ rossEntropy(ˆy, y)   
15: gradients for $\Theta , \alpha , \beta$ by autograd $\{ \theta , \phi$ held at initial values}   
16: update $\Theta , \alpha , \beta$ with Adam

## D. Experimental Design and Evaluation Metrics

The design principle throughout is that a quantum model is compared against the classical model it corresponds to (same features, same folds, same optimiser, same training budget) rather than against the best classical number available anywhere; Section X-E describes the artefacts this matching prevents.

The evaluation protocol follows three rules.

Leave-site-out (LSO) validation: the dataset is split by acquisition site. In each fold, one site is held out entirely for testing while the rest train the model, which reduces the ability to exploit site-specific scanner characteristics, a wellknown confound in ABIDE-based work. UCLA-CNP, with two scanners, is the exception (Section IV-A).

Baselines: linear classical (support vector machine (SVM) with a linear kernel; graph convolutional network (GCN) [32] with a Pearson-correlation graph) and non-linear classical (SVM with an RBF kernel; GCN with an RBF-kernel similarity graph), with both SVMs applied to flattened correlation matrices (Tables II–IV).

Statistical testing: every quantum and classical pair is compared with a paired Wilcoxon signed-rank test over identical folds, with 95% confidence intervals (CIs) across folds reported for the ABIDE I results of Tables II–IV and Bonferroni correction applied wherever a sweep of comparisons is reported. A label-permutation null is run for the kernel arm on ABIDE I, shuffling labels within site so that per-site class prevalence is preserved and feature selection, scaling, and bandwidth choice are repeated inside every fold on the permuted labels (Section V-B).

Algorithm 2 Matched comparison of a quantum model and   
its classical twin   
Require: cohort D with sites $1 , \ldots , S ;$ quantum model Q and   
classical model $C$ matched on features, capacity, optimiser and   
training budget   
Ensure: per-fold metrics, paired p-values, screened wins   
1: for each site s carrying both classes {UCLA-CNP: stratified 10-   
fold} do   
2: $\tilde { \mathcal { D } } _ { \mathrm { t r } }  \mathcal { D } \setminus s ; \ \mathcal { D } _ { \mathrm { t e } }  s$   
3: $s $ selec $( \mathcal { D } _ { \mathrm { t r } } , n ) .$ , P ← basis $( \mathcal { D } _ { \mathrm { t r } } ) ;$ apply $P \circ S$ unchanged   
to $\mathcal { D } _ { \mathrm { t e } }$ {no fit sees $\left. \mathcal { D } _ { \mathrm { t e } } \right\}$   
4: kernel models: $\gamma ^ { \ast } \gets$ arg max<sub>γ</sub> $\mathrm { K T A } \big ( K _ { \gamma } ( \mathcal { D } _ { \mathrm { t r } } ) , y y ^ { \top } \big )$   
{Eq. (11), on ${ \mathcal { D } } _ { \mathrm { t r } } \}$   
5: train Q and $C$ on ${ \mathcal { D } } _ { \mathrm { t r } }$ with the same seed and budget   
6: record accuracy, F1, AUC, sensitivity, specificity and MCC of   
both on $\mathcal { D } _ { \mathrm { t e } }$   
7: end for   
8: for each metric $m$ do   
9: $p _ { m } \gets$ paired Wilcoxon signed-rank test on the per-fold   
differences $m ( Q ) - m ( C )$   
10: end for   
11: mark Q or C degenerate if, pooled over folds, its sensitivity or   
specificity is below 0.30 or $| \mathrm { \hat { M } C C } | < 0 . 0 2$   
12: report each $p _ { m }$ uncorrected and against $0 . 0 5 / M$ for the M tests   
of the sweep; count a win as clean only if $p _ { m } < 0 . 0 5$ and neither   
Q nor $C$ is degenerate

Alongside accuracy, F1, AUC, and specificity, we report the Matthews correlation coefficient (MCC)

$$
\mathrm { M C C } = \frac { T P \cdot T N - F P \cdot F N } { \sqrt { \hat { P } P \hat { N } N } } ,\tag{10}
$$

where TP, TN, FP and FN count true positives, true negatives, false positives and false negatives, with $P = T P { + } F N ,$ $N ~ = ~ T N / { \cal F P } , ~ \hat { \cal P } ~ = ~ T { \cal P } / + F { \cal P }$ and $\hat { N } \ = \ T N { + } F N ,$ which, unlike accuracy, F1 or specificity, stays near zero for a predictor that assigns every subject to one class. Algorithm 2 sets out the matched protocol that every quantum–classical comparison follows.

## V. EXPERIMENTAL RESULTS

## A. Classification Performance

Tables II–IV report performance averaged across the 20 ABIDE I site folds. Holding out an entire scanner lowers absolute accuracy relative to a random split but estimates generalisation to an unseen site. The three tables answer different questions and are not interchangeable: graph-based models build a topology over 200 regions and classify the graph, feature-based models (Table III; defined in Section VI) receive the eight supervised-selected connectivity features that the register admits (Section IV-A), and the reference models take the full correlation vector with no register limit. F1 and specificity treat the disorder group as positive. MCC is the column that separates classification from degeneracy: it stays at zero for a predictor that assigns every subject to one class, where F1 reaches ≈ 0.65 on a near-balanced cohort.

Table II  
LEAVE-SITE-OUT PERFORMANCE OF THE GRAPH MODELS ON ABIDE I(1,035 SUBJECTS, 20 SITE FOLDS, EIGHT QUBITS; MEAN ± 95% CI,NORMAL QUANTILE).
<table><tr><td>Model</td><td>F1</td><td>Acc</td><td>Spec</td><td>MCC</td></tr><tr><td>GCN+Pearson</td><td>0.252±0.118</td><td>0.494±0.029</td><td>0.664±0.185</td><td>-0.007±0.036</td></tr><tr><td>GCN+RBF</td><td>0.145±0.119</td><td>0.508±0.029</td><td>0.783±0.179</td><td>-0.009±0.017</td></tr><tr><td>Quantum+GAT</td><td>0.196±0.135</td><td>0.504±0.028</td><td>0.700±0.206</td><td>0.000±0.000</td></tr><tr><td>Quantum+GAT, trained enc.</td><td>0.237±0.130</td><td>0.513±0.036</td><td>0.705±0.174</td><td>0.010±0.055</td></tr><tr><td>Quantum+GCN (abl.)</td><td>0.133±0.119</td><td>|0.512±0.027</td><td>0.800±0.180</td><td>0.000±0.000</td></tr></table>

For contrast, the unrestricted linear SVM of Table IV reaches F1 0.653, accuracy 0.671 and MCC 0.346 on the same folds.

Table III  
MATCHED EIGHT-FEATURE COMPARISONS ON ABIDE I, EACH QUANTUM MODEL (†) ABOVE ITS CLASSICAL TWIN (MEAN ± 95% CI OVER 20 SITE FOLDS).
<table><tr><td>Model</td><td>Acc</td><td>AUC</td><td>Spec</td><td>MCC</td></tr><tr><td>QCNN† CNN (1-D)</td><td>0.615±0.029 0.589±0.038</td><td>0.666±0.036</td><td>0.699±0.076</td><td>0.253±0.056</td></tr><tr><td>VQC†</td><td>0.582±0.032</td><td>0.638±0.043 0.612±0.043</td><td>0.652±0.074 0.632±0.074</td><td>0.194±0.075 0.179±0.066</td></tr><tr><td>MLP</td><td>0.600±0.028</td><td>0.629±0.037</td><td>0.674±0.055</td><td>0.208±0.057</td></tr><tr><td>QSVM-fixed†</td><td>0.622±0.034</td><td>0.663±0.041</td><td>0.717±0.076</td><td>0.264±0.070</td></tr><tr><td>SVM (RBF), matched</td><td>0.616±0.030</td><td>0.662±0.039</td><td>0.702±0.075</td><td>0.255±0.059</td></tr><tr><td>TQEK†</td><td>0.610±0.036</td><td>0.686±0.036</td><td>0.724±0.069</td><td>0.234±0.077</td></tr><tr><td>Trainable-RBF</td><td>0.581±0.036</td><td>0.606±0.044</td><td>0.586±0.062</td><td>0.177±0.074</td></tr></table>

Table IV

UNRESTRICTED CLASSICAL REFERENCES ON ABIDE I: FULL CORRELATION VECTOR, NO REGISTER LIMIT.
<table><tr><td>Model</td><td>F1</td><td>Acc</td><td>Spec</td><td>MCC</td></tr><tr><td>SVM (Linear), full corr.</td><td>0.653±0.039</td><td>0.671±0.036</td><td>0.689±0.067</td><td>0.346±0.073</td></tr><tr><td>SVM (RBF), full corr.</td><td>0.627±0.054</td><td>0.642±0.042</td><td>0.634±0.088</td><td>0.305±0.084</td></tr></table>

Three observations emerge. First, the largest performance difference was observed between graph-based and correlationfeature classifiers rather than between quantum and classical methods: both SVMs separate the classes (0.671 and 0.642 accuracy), while all five graph configurations sit at chance (0.494 to 0.513), and the quantum convolutional network reaches 0.615 against 0.589 for its matched classical counterpart. Second, additional kernel non-linearity did not improve performance under the evaluated feature representations and validation protocol: the linear SVM beats the RBF SVM (0.653 against 0.627 F1), and GCN on Pearson beats GCN on RBF (0.252 against 0.145). Third, quantum topology did not outperform correlation topology under the evaluated datasets and experimental settings: Quantum+GAT reaches F1 0.196 against 0.252 for GCN+Pearson, with accuracies within 0.010 of each other and of chance. Specificity is the only column where the quantum configurations appear competitive, but the highest specificities belong to the worst classifiers: the quantum-GCN ablation attains 0.800 with an F1 of only 0.133. Per-fold inspection shows folds collapsing onto a single predicted class, which the MCC column states directly: both fidelity-graph configurations sit at 0.000 ± 0.000, and only the correlation-based references clearly exceed zero.

Supervised selection is consequential in two ways. Relative to PCA, held-out AUC on ABIDE I rises from 0.729 to 0.738, and, more importantly, the quantum comparison changes character: what appeared to be a significant quantum deficit under PCA reaches statistical parity with the classical result once the features are chosen for discriminability.

![](images/3c03d5d51273133392871f98f4e48ff07ec14902cda26bd6ff51172756d66dec.jpg)  
Figure 4. Quantum kernel against classical kernels given matched and full feature sets on four cohorts.

## B. Matched-Feature Comparison

A quantum kernel restricted to n qubits sees n features, while an unrestricted classical model sees 2,000. Comparing them directly measures dimensionality rather than the kernel. We therefore add a matched control: a classical RBF kernel, $k _ { \gamma } ( x , x ^ { \prime } ) = \exp ( - \gamma \| x - x ^ { \prime } \| ^ { 2 } )$ with width parameter γ, given exactly the n<sub>qubits</sub> features the quantum kernel receives.

Under the matched control, the quantum kernel and the classical kernel are statistically indistinguishable on every cohort: ABIDE I 9/20 folds (p = 0.51), ADHD-200 2/7 (p = 0.56), REST-meta-MDD 10/25 (p = 0.59), UCLA-CNP 4/10 (p = 0.25). Across 62 folds, no statistically significant differences were observed. Section X-C bounds how large any difference can be. Fig. 4 shows the quantum kernel tracking a classical kernel given the same eight features, while both trail a classical model given all 2,000, with each cohort’s fold wins and paired p-value marked above its bars.

These findings suggest that the quantum-edge deficit in the population-graph comparison of Section VII was an artefact of comparing eight quantum features against 128-dimensional classical descriptors. Where the quantum kernel does lose to the full 2,000-feature models on ABIDE and REST-meta-MDD, the cause is the register limit rather than the kernel.

## C. Permutation, Seed and Noise Robustness Checks

Three checks establish that the parity is between working models rather than between two failures, and that it does not depend on noiseless simulation. First, a label-permutation null: labels were shuffled within site, preserving each site’s class prevalence, and the leave-site-out sweep was re-run 104 times with feature selection, scaling, and bandwidth choice repeated inside every fold. Every observed accuracy and AUC exceeded all 104 permutations, so the empirical p reaches its floor of 0.0095, below the Bonferroni threshold of 0.0125 for four models, and the nulls sit at chance to three decimals. QSVMfixed reaches 0.622 accuracy and 0.663 AUC against nulls of 0.498 ± 0.019 and $0 . 4 9 9 \pm 0 . 0 2 2 ;$ ; TQEK 0.610 and 0.686 against $0 . 4 9 9 \pm 0 . 0 2 4$ and $0 . 4 9 8 \pm 0 . 0 3 2 ;$ the matched SVM (RBF) 0.616 and 0.662 against $0 . 4 9 9 \pm 0 . 0 2 5$ and 0.498 ± 0.030; and Trainable-RBF 0.581 and 0.606 against 0.500 ± 0.017 and $0 . 4 9 6 \pm 0 . 0 2 2$

Second, the suite was repeated at four seeds. The largest standard deviation of mean accuracy is 0.017, and the two kernel models without a stochastic component are identical at every seed. Three of the four matched differences hold their sign throughout (QCNN +0.025, TQEK +0.026, QSVM-fixed +0.006 over their twins); only VQC against the matched MLP changes sign. The consistency is in direction, not magnitude: none survives the correction of Section X-B, and the largest is smaller than the cost of the eight-feature budget itself.

Third, shot and device noise: with every kernel entry estimated from a finite number of measurement shots on the same ABIDE I folds, QSVM-fixed reaches 0.593 accuracy at 128 shots and 0.622–0.643 (AUC 0.654–0.667) from 256 to 8,192 shots, against 0.622 under exact simulation; a global depolarising channel of rate 0.05–0.4 leaves it at 0.621–0.634 (AUC 0.652–0.669). No noisy setting falls significantly below the matched RBF kernel.

## D. Why the Two Kernels Perform Alike

A Gram matrix’s participation ratio, $( \sum _ { i } \lambda _ { i } ) ^ { 2 } / \sum _ { i } \lambda _ { i } ^ { 2 }$ , with $\lambda _ { i }$ the eigenvalues of the matrix, counts how many directions a kernel actually uses, and we refer to it as the effective dimension. Near 1, the kernel is almost constant and every subject looks alike; near the number of subjects, it approaches the identity and every subject becomes its own cluster. Useful kernels lie in between.

Measured on 400 ABIDE subjects with eight selected features, both kernels sweep the same range as their width parameter varies, from an effective dimension of 1.1 to 343 out of a possible 400 (Fig. 5, blue). Both possess a usable window: quantum fidelity at bandwidth 0.15 reaches effective dimension 10.8, against 8.2 for the RBF kernel at $\gamma = 0 . 1 2 5$ At that matched setting, their eigenvalue spectra are nearly indistinguishable: the logarithms of the leading 40 normalised eigenvalues correlate at $r ~ = ~ 0 . 9 9 7$ , and those eigenvalues carry 87.0% and 87.9% of the total spectral weight. The same overlap through the usable window holds on all four cohorts, as the remaining curves of Fig. 5 show.

This provides a mechanistic account of the matched performance in Fig. 4. The quantum feature map defines a nonlinear similarity function that, on these data, behaves like a bandwidth-controlled RBF kernel: the two kernels span the same range of effective dimension, their leading eigenspectra nearly coincide once matched, and the encoding bandwidth of the quantum kernel plays the role that γ plays for the RBF kernel. This empirical similarity does not prove that the two kernels belong to the same mathematical family, or that one is a reparameterisation of the other; it does, however, offer a plausible explanation for their comparable predictive performance.

It also clarifies why bandwidth was the decisive hyperparameter (Section V-B). An untuned bandwidth of 1.0 places the quantum kernel at effective dimension 289, close to the identity, where it memorises the training set and generalises poorly, which is where an unreported bandwidth would leave it by default.

![](images/eea9a172867e2e4327ce3cd6ff8cc5694d5bceafe72e8805403b55d170652714.jpg)  
Figure 5. Effective dimension of the quantum fidelity and classical RBF kernels against their width parameter on all four cohorts.

## VI. QUANTUM MODEL SUITE

The matched comparisons of Section V-B turn on a single quantum construct, a fidelity kernel. To test whether the conclusion is specific to that choice, we evaluated five quantum models drawn from the Qiskit Machine Learning reference implementations, each against the classical model it is naturally matched to, on identical features and identical folds.

## A. Models

QCNN follows the quantum convolutional network of Cong et al. [33] as given by the Qiskit tutorial: a ZFeatureMap encoding, alternating convolution and pooling layers, and a Pauli-Z read-out on the surviving qubit. We reimplemented it in a batched statevector simulator because Qiskit’s EstimatorQNN evaluates one sample per circuit execution, which does not scale to 2,428 subjects across 25 folds; equivalence to the reference is verified to 10<sup>−6</sup>.

VQC is the standard variational quantum classifier: ZZFeatureMap encoding [21], a RealAmplitudes ansatz, and a parity read-out mapping the measured basis distribution onto two classes. Unlike the ring encoder of Section IV-B, which encodes the input as a rotation angle and returns n expectation values, it encodes the input as a phase, once per qubit and again per entangled pair, and collapses the $2 ^ { n } .$ -dimensional register onto a single parity bit. A phase of $2 x$ is 2π-periodic, so a feature scaled to [0, π] maps its two extremes onto the same state; both variational arms are therefore scaled to $[ 0 , \pi / 2 ]$ , which keeps every feature value on a distinct state.

QSVM-fixed is the fidelity-kernel quantum support vector machine (QSVM) [34] of Section V-B, with bandwidth selected by kernel-target alignment. QSVM-Pegasos applies Qiskit’s PegasosQSVC to the same precomputed kernel.

TQEK is a trainable quantum embedding kernel [35], in which the embedding’s parameters are optimised against kernel-target alignment so the feature geometry adapts to the task. Section V-D motivates it directly: a fixed kernel has one free parameter in its width and therefore traverses the same family of Gram matrices as an RBF kernel, whereas a trainable embedding does not.

![](images/27965271ff8cbcd4f9085b86da6c07ee979b47c5cddf51cab00590d4da227d7b.jpg)  
Figure 6. Median per-fold accuracy and AUC difference between each quantum model and its matched classical twin, for the model suite and the three interventions.

Each quantum model is paired with a classical comparator of matched capacity and identical training budget. TQEK’s comparator is an RBF kernel on a learned linear metric, optimised under the same objective, optimiser and step count. Without it, a trained quantum kernel beating an untrained classical one would demonstrate only that training helps.

## B. Results

Table V reports every model on every cohort across five metrics. Quantum models, marked †, and their matched comparators see identical features and identical folds; SVM rows labelled “2,000 feat.” receive the full 2,000 supervisedselected features, the population-graph rows (quantum population graph, QPG; classical population graph, PG; and the no-graph MLP) use the 128-dimensional descriptors of Section VII, and PegasosQSVC reports no AUC. Fig. 6 shows the median per-fold difference in accuracy and AUC between every quantum model and its matched twin, for this suite and for the interventions of Section IX, with $p < 0 . 0 5$ on a paired Wilcoxon test marked; the differences quoted below are these medians.

Twenty paired accuracy comparisons were made. Two reach $p \ < \ 0 . 0 5$ uncorrected, both on ABIDE I and in opposite directions: TQEK beats its comparator $( + 0 . 0 2 8 , p = 0 . 0 4 9 )$ while QSVM-Pegasos loses to its own $( - 0 . 0 2 9 , p = 0 . 0 3 9 )$ With twenty tests, one expects roughly one false positive at $\alpha = 0 . 0 5 ;$ the inconsistent effect directions and the failure of both to survive Bonferroni correction (0.05/20 = 0.0025) are compatible with sampling variation.

TQEK is the only quantum model with a significant accuracy win anywhere in this suite, and it does not replicate: $+ 0 . 0 2 8 \ ( p = 0 . 0 4 9 )$ on ABIDE I but $- 0 . 0 1 3 \ ( p = 0 . 2 7 )$ on REST-meta-MDD and −0.045 (p = 0.088) on UCLA-CNP. Section VII shows similar cohort-dependent behaviour.

QCNN is the most consistent quantum model in the suite: it leads its matched classical convolutional network on ABIDE I (+0.044) and UCLA-CNP (+0.023), matches it on RESTmeta-MDD, and is also ahead on ADHD-200 (+0.030), where that comparator does not separate the classes (Section X-B). The ABIDE I difference approaches significance $( p = 0 . 0 5 8 )$ indicating a consistent direction of effect, and QCNN is the only quantum model that does not trail its comparator on any cohort.

PegasosQSVC exposes no decision function, so no AUC is available, and on ADHD-200 it predicts a single class on every fold $( \mathrm { F 1 } ~ = ~ 0 . 0 0 0$ at the majority-class accuracy of 0.561). Runtimes favour the classical side by up to two orders of magnitude: on REST-meta-MDD, QSVM-fixed takes 0.67 s per fold against 0.06 s for SVM (RBF), TQEK 3.0 s against 0.10 s for Trainable-RBF, VQC 15.2 s against 0.07 s for the MLP, and QCNN 32.6 s against 8.6 s for the classical CNN, on an exact simulator where the quantum models incur no shot noise or device overhead.

## VII. MULTI-COHORT VALIDATION

The ABIDE results suggest performance parity under matched feature constraints. To examine whether this observation generalises across disorders and preprocessing pipelines, we applied the same framework to three further public cohorts (Table I). All design choices were fixed on ABIDE I before these cohorts were analysed, so they provide an independent, out-of-sample evaluation.

## A. Quantum Kernel Versus Classical Kernels

We evaluate the subject-level quantum kernel $\begin{array} { r l } { K _ { i j } } & { { } = } \end{array}$ $| \langle \psi ( x _ { i } ) | \psi ( x _ { j } ) \rangle | ^ { 2 }$ (here $x _ { i }$ is the descriptor of subject i) against classical kernels on identical descriptors and identical folds. Encoding bandwidth and kernel type are selected within each fold by kernel-target alignment,

$$
\mathrm { K T A } ( K , y ) = \frac { \langle K , y y ^ { \top } \rangle _ { F } } { \| K \| _ { F } \| y y ^ { \top } \| _ { F } } ,\tag{11}
$$

where K is the Gram matrix of the m training subjects, $y \in$ $\{ - 1 , + 1 \} ^ { m }$ holds their labels, and $\langle \cdot , \cdot \rangle _ { F }$ and $\| \cdot \| _ { F }$ are the Frobenius inner product and norm. The classical comparator additionally optimises its width parameter, providing at least comparable tuning flexibility. Because both models see the same folds, significance is assessed with a paired Wilcoxon signed-rank test over per-fold scores.

A second framing places subjects rather than regions on the graph. In this population graph, each node is a subject carrying a 128-dimensional connectivity descriptor, edges come from the eight-qubit fidelity kernel (QPG) or from classical RBF or correlation kernels (PG), and an identical GCN classifies every variant; an MLP given the same descriptors with no graph is the reference. On ABIDE I, the quantum edges reach an accuracy of 0.537, against 0.589 and 0.601 for correlation and RBF edges and 0.662 for the no-graph MLP.

## B. Evaluation of Replication Across Cohorts

(a)

On UCLA-CNP, the quantum kernel outperforms the classical linear kernel on 8 of 10 folds $( p \ : = \ : 0 . 0 3 7 )$ , whereas on REST-meta-MDD the same comparison favours the linear kernel, which wins on 18 of 25 folds $( p = 0 . 0 1 1 )$ .

REST-meta-MDD is the stronger evidence, with roughly ten times as many subjects and folds that hold out whole sites, so the UCLA-CNP result more likely reflects cohortspecific variability than a reproducible effect. Three further observations support this reading: on UCLA-CNP the AUC of every model in this comparison sits at chance (0.490–0.514); the advantage appears only against the linear kernel and never against the RBF kernel on any cohort; and of the six paired tests across the three further cohorts, the two that reached $p \ < \ 0 . 0 5$ point in opposite directions and neither survives Bonferroni correction $( 0 . 0 5 / 6 = 0 . 0 0 8 )$

On REST-meta-MDD the full-dimensional classical RBF reaches $\mathrm { F 1 = 0 . 6 7 1 }$ and $\mathrm { A U C } = 0 . 6 8 4$ , against 0.571 and 0.544 for the quantum kernel on eight principal components.

## VIII. LIMITS OF THE FIDELITY GRAPH

This section examines two properties of fidelity-based graph construction that hold beyond our implementation (the register width at which fidelity still supplies usable edges, and an algebraic invariance that fixes which circuit parameters enter it) and then reports the graph pipeline’s results.

## A. Register Width Determines Whether Fidelity Is Usable

Fidelity remains useful for graph construction only while sufficient numbers of region pairs exhibit measurable overlap. Measuring all 19,900 region pairs on twenty subjects, with each region described by the connectivity profile the graph pipeline uses, shows how rapidly pairwise fidelity decreases with increasing register width. Fig. 7(a) plots the mean and maximum pairwise fidelity on a log scale against a reference floor of 0.01, with the share of region pairs above that floor as bars. From four qubits to eight, twelve and sixteen, the Hilbert dimension grows from 16 to 256, 4,096 and 65,536; mean fidelity falls from 0.0652 to 0.0083, 0.0037 and 0.0027, crossing the floor between four and eight qubits; and the share of pairs above the floor falls from 48.4% to 12.6%, 6.1% and 4.4%.

What matters is not the share but whether enough pairs survive to build the graph. $\textbf { A } k = 2 0$ graph over 200 regions needs 4,000 directed edges, spread over at least 2,000 distinct pairs. At eight qubits, a median of 2,336 pairs clear the floor, but unevenly, so 11% of a fidelity-only graph’s edges already fall below it; at twelve and sixteen qubits only 1,108 and 787 pairs clear it, and 48% and 61% of the edges are taken from below-floor similarities. The floor is a convention, and the crossover moves with it: at 0.003 twelve qubits still supply 2,246 pairs, at 0.001 even sixteen supply 2,853, and at 0.03 eight qubits supply only 1,169. What holds at every floor is the direction: each added qubit leaves fewer pairs, and a smaller share of the graph’s edges, above it. The parcellation supplies 200 regional states; a 65,536-dimensional space has far more room than 200 states can populate, so the typical overlap falls toward zero. This is the concentration effect established for quantum kernels [36], though on structured connectivity features it is far slower than for random states: mean fidelity at sixteen qubits is 0.0027, about 180 times the $2 ^ { - 1 6 } \approx 1 . 5 \times 1 0 ^ { - 5 }$ of random states. It shares a root with barren plateaus [37], [38], namely measure concentration in an exponentially large Hilbert space, but the two are distinct: concentration makes kernel values indistinguishable, and bites even where no gradient is taken, whereas a barren plateau flattens a variational gradient. Section VIII-B separates them here.

![](images/a96d03d264062e9b45ab67bfacc46e7cae3439b47bd8f3d215ad9c4dba61daee.jpg)  
qubits per region (Hilbert dimension)

![](images/2d313093df8ff2bd9503d56d98a175dcc48c2af79a656b45a2c0673d3b54f522.jpg)  
Figure 7. Fidelity concentration with register width over twenty ABIDE I subjects, and one subject’s connectivity under Pearson correlation and fourand sixteen-qubit fidelity.

This concentration limits the usefulness of wider registers for the evaluated fidelity graph. A graph remains usable under near-orthogonality only while enough region pairs stay nonorthogonal; once the bulk of the matrix is orthogonal, too few pairs are left to fill $\mathrm { ~ a ~ } k = 2 0$ neighbourhood, and the selection step must draw an increasing proportion of edges from below-floor similarities, whose reliability as connectivity estimates has not been established. On representative subjects, the Pearson matrix is dense (average degree $k \approx 8 0 )$ and fourqubit fidelity markedly sparser $( k \approx 2 5$ after thresholding); for one subject the mean absolute value falls from 0.394 under Pearson correlation to 0.063 at four qubits and 0.002 at sixteen, where the matrix is sparse apart from its diagonal (Fig. 7(b)). The practical consequence is an operating rule, not a bound. Its general half is arithmetic: a k-nearest-neighbour graph over N regions needs kN edges on at least $k N / 2$ distinct pairs, and each region needs k above-floor neighbours of its own, so the pair count is necessary but not sufficient. Its other half is empirical, since the crossover depends on the encoder, the parcellation, the feature parameterisation and the floor itself; on these data, at $N = 2 0 0$ and $k = 2 0$ , the widest register that still supplies enough pairs falls from sixteen qubits at a floor of 0.001 to four at 0.03, far short of one qubit per feature. We read $2 ^ { n } \gg N$ as a summary of that measurement, not a derived limit.

B. Register Width, Accuracy and Computational Cost in Variational Models

Section VIII-A concerns kernels, which compare states pairwise. A QCNN reads a single Pauli-Z expectation instead, so concentration need not bite the same way, and whether width helps a variational model was an open question rather than a measured one.

Training a QCNN at four, eight and sixteen qubits on a 300-subject pool with a held-out site gives test accuracies of 0.491, 0.503 and 0.520, with the overfit gap narrowing from +0.149 to +0.110. The present experiment suggests that fidelity concentration does not directly translate into degraded QCNN performance. However, the observed performance gain was achieved at substantial computational cost: sixteen qubits cost 163× the runtime of eight (685.5 s against 4.2 s) for +0.017 accuracy, and statevector memory doubles per qubit.

Training accuracy never exceeds 0.65 at any width, so the model underfits while simultaneously overfitting by 0.11 to 0.15. This is a single-site diagnostic rather than a crossvalidated result, and no accuracy here should be read as a cohort figure; Section IX-C tests the remaining option, adding capacity at fixed width.

## C. The Variational Layer Is Invisible to the Graph

A third limitation is algebraic rather than empirical: it constrains what the fidelity graph can learn at any register width.

Our encoder prepares $| \psi ( x ) \rangle = U V ( x ) | 0 \rangle$ , where $V ( x )$ applies the trainable encoding gains and U is the entangling ring followed by the variational rotations. U is unitary and applied identically to every region, so it cancels exactly in the fidelity, because $U ^ { \dagger } U = I { \mathrm { : } }$

$$
F _ { i j } = { \big | } \langle 0 | V ( x _ { i } ) ^ { \dagger } U ^ { \dagger } U V ( x _ { j } ) | 0 \rangle { \big | } ^ { 2 } = { \big | } \langle 0 | V ( x _ { i } ) ^ { \dagger } V ( x _ { j } ) | 0 \rangle { \big | } ^ { 2 } .\tag{12}
$$

Everything after the data-dependent encoding leaves the fidelity kernel invariant and receives no gradient from the graph. We confirmed this numerically: perturbing the variational parameters by O(1) changes the fidelity Gram matrix by $1 . 2 \times 1 0 ^ { - 6 }$ , and their gradients through the graph are $1 . 0 \times 1 0 ^ { - 7 }$ and $3 . 8 \times 1 0 ^ { - 8 }$ against 51.5 for the encoding gains. Through the Pauli-Z read-out path the same parameters receive gradients of 0.128 and 0.298, so the circuit is not inert. It is only its contribution to the graph that vanishes.

The encoding gains are therefore the only circuit parameters that enter a fidelity graph: eight parameters for a 200-node topology, rather than the full variational stack the architecture appears to provide. In the principal graph runs, the encoder is evaluated once, without gradients, at its initial parameters, and the top-k support selected from that encoding stays fixed while the edge kernel and the classifier are optimised; training the encoder as well, with the support re-selected every epoch, leaves the graph at chance (Section VIII-D). The invariance therefore does not affect any reported result; it constrains how a fidelity graph can be trained.

The argument extends to any architecture in which identical post-encoding unitary operations are applied before fidelity is computed. In such constructions, only trainable parameters that precede or are interleaved with the data-dependent encoding can influence the fidelity; a unitary block placed after the final encoding layer and applied identically to every input cancels regardless of its depth. Even when every parameter enters the fidelity, the support is fixed by a top-k selection that carries no gradient, so a fidelity graph is trainable in its edge weights but not in its topology.

The identity reaches beyond graphs to any hybrid model that compares data by state overlap, including fidelity kernels in support vector machines and kernel-alignment training whose optimised block follows the encoding. In each case, the kernel’s gradient with respect to those layers is exactly zero at every parameter setting, unlike the exponentially small gradients of barren plateaus [37], [38], so barren-plateau mitigations such as tailored initialisation [39] or layer-wise training [40] cannot recover it, while the cancelled layers still cost depth, simulation time and, on hardware, gate noise. Trainable similarity therefore has to sit where it survives: inside the feature map, as in trainable embedding kernels [35]; between repeated encodings, as in data re-uploading [41], where only the last trainable block cancels; or in classical layers applied after the overlap, such as graph attention. Perturbing the post-encoding parameters and checking whether the Gram matrix moves is enough to tell which case a published hybrid falls into.

## D. Performance Limitations of the Fidelity-Graph Pipeline

The graph-attention pipeline stays at chance in every config uration we tried, with accuracies of 0.504 for the fidelity graph with attention, 0.512 for its GCN ablation and 0.513 with a trained encoder, and in almost every fold it assigns the whole held-out site to one class (MCC 0.000 for both fidelity-graph models; Table II). Modifying the kernel, changing the readout, supplying node features that a linear classifier separates at AUC 0.739, and training the encoder jointly while reselecting the support every epoch did not materially improve performance, and a quantum graph transformer behaved the same way. Because informative node features did not help, the remaining candidate explanations include sparsification effects, the quality of the fidelity-selected topology, limitations of attention learning and graph over-smoothing; the present study does not isolate their individual contributions, so we report the graph arm as a negative result, and every quantum– classical comparison in this paper rests on the feature-based models.

## IX. ARCHITECTURAL INTERVENTIONS ON THE FEATURE BUDGET

Section V-B located the deficit in the register’s feature budget rather than in the kernel: at eight features the quantum and classical kernels are indistinguishable, while a classical model given all 2,000 features is better. We therefore designed three interventions that lift that budget without widening the register, each reported with matched classical twins receiving the identical construction, because an intervention that helps only when the classical arm is denied it measures the comparison rather than the model. Rows A–C of Fig. 6 summarise the matched comparisons.

## A. A Learned Projection

Fixed t-test selection is replaced by a linear map $\mathbb { R } ^ { 2 0 0 0 } $ R<sup>8</sup> learned jointly with the circuit, so the quantum stage receives an optimised summary of every connection rather than eight raw ones. Every quantum head is given a classical twin with an identical projection, identical bottleneck width and identical training, plus a linear-head control that measures how much of any gain comes from the projection alone. The prefix H- marks these projection-fed heads.

The projection helps the classical arm and leaves the convolutional quantum head statistically indistinguishable from its classical twins. On ABIDE I, H-QCNN reaches 0.645, the best model in the projection comparison, against 0.641 for H-CNN and 0.630 for H-Linear; neither comparison approaches significance $( p \ = \ 0 . 3 7$ and $ { p ^ { \mathrm { ~ ~ } } } = \ 0 . 1 8 )$ . On REST-meta-MDD it reaches 0.573 against 0.589 and 0.591, again with no significant separation $( p = 0 . 2 7 , p = 0 . 2 5 )$

Table V  
EVERY MODEL ON EVERY COHORT. ALL VALUES ARE MEANS OVER FOLDS; LEAVE-SITE-OUT EXCEPT UCLA-CNP, WHICH IS STRATIFIED 10-FOLD. † MARKS A QUANTUM MODEL. SVM LABELS GIVE EACH MODEL’S INPUT: EIGHT MATCHED FEATURES, OR THE 2,000 SUPERVISED-SELECTED ONES.
<table><tr><td>Model</td><td>Acc</td><td>F1 AUC</td><td></td><td>Spec</td></tr><tr><td>ABIDE I</td><td></td><td></td><td></td><td>MCC</td></tr><tr><td>QCNN†</td><td>0.615</td><td>0.561 0.666</td><td>0.699</td><td>0.253</td></tr><tr><td>QPG (quantum edges)†</td><td>0.537 0.528</td><td>0.576</td><td>0.514</td><td>0.074</td></tr><tr><td>QSVM-Pegasos†</td><td>0.579 0.475</td><td></td><td>0.750</td><td>0.199</td></tr><tr><td>QSVM-fixed†</td><td>0.622 0.560</td><td>0.663</td><td>0.717</td><td>0.264</td></tr><tr><td>TQEK↑</td><td>0.610 0.534</td><td>0.686</td><td>0.724</td><td>0.234</td></tr><tr><td>VQC†</td><td>0.582 0.535</td><td>0.612</td><td>0.632</td><td>0.179</td></tr><tr><td>CNN (1-D)</td><td>0.589 0.538</td><td>0.638</td><td>0.652</td><td>0.194</td></tr><tr><td>MLP</td><td>0.600 0.550</td><td>0.629</td><td>0.674</td><td>0.208</td></tr><tr><td>MLP (no graph)</td><td>0.662 0.641</td><td>0.729</td><td>0.688</td><td>0.335</td></tr><tr><td>PG (RBF edges)</td><td>0.601 0.589</td><td>0.630</td><td>0.593</td><td>0.211</td></tr><tr><td>PG (correlation edges)</td><td>0.589 0.587</td><td>0.639</td><td>0.574</td><td>0.195</td></tr><tr><td>SVM (RBF), 8 feat.</td><td>0.616 0.559</td><td>0.662</td><td>0.702</td><td>0.255</td></tr><tr><td>SVM (RBF), 2,000 feat.</td><td>0.649 0.632</td><td>0.738</td><td>0.642</td><td>0.323</td></tr><tr><td>SVM (Linear), 2,000 feat.</td><td>0.629 0.624</td><td>0.692</td><td>0.618</td><td>0.268</td></tr><tr><td>Trainable-RBF</td><td>0.581 0.562</td><td>0.606</td><td>0.586</td><td>0.170</td></tr><tr><td>ADHD-200</td><td></td><td></td><td></td><td></td></tr><tr><td>QCNN†</td><td>0.512 0.408</td><td>0.551</td><td>0.609</td><td>0.019</td></tr><tr><td>QPG (quantum edges)†</td><td>0.526 0.363</td><td>0.558</td><td>0.694</td><td>0.026</td></tr><tr><td>QSVM-Pegasos†</td><td>0.561 0.000</td><td></td><td>0.998</td><td>-0.009</td></tr><tr><td>QSVM-fixed†</td><td>0.546 0.458</td><td>0.561</td><td>0.625</td><td>0.105</td></tr><tr><td>TQEK†</td><td>0.549 0.421</td><td>0.551</td><td>0.674</td><td>0.074</td></tr><tr><td>VQCt</td><td>0.503 0.426</td><td>0.538</td><td>0.544</td><td>0.015</td></tr><tr><td>CNN (1-D) MLP</td><td>0.495 0.385</td><td>0.504</td><td>0.609</td><td>0.018</td></tr><tr><td></td><td>0.540 0.437</td><td>0.548</td><td>0.619</td><td>0.060</td></tr><tr><td>MLP (no graph) PG (RBF edges)</td><td>0.535 0.234</td><td>0.522</td><td>0.824</td><td>-0.014</td></tr><tr><td>PG (correlation edges)</td><td>0.524 0.308</td><td>0.493</td><td>0.730</td><td>-0.001</td></tr><tr><td>SVM (RBF), 8 feat.</td><td>0.503 0.285</td><td>0.495</td><td>0.726</td><td>-0.013</td></tr><tr><td></td><td>0.571 0.448</td><td>0.573</td><td>0.683</td><td>0.125</td></tr><tr><td>SVM (RBF), 2,000 feat.</td><td>0.561 0.264</td><td>0.555</td><td>0.841</td><td>0.034</td></tr><tr><td>SVM (Linear), 2,000 feat.</td><td>0.508 0.331</td><td>0.489</td><td>0.702</td><td>0.006</td></tr><tr><td>Trainable-RBF REST-meta-MDD</td><td>0.562 0.430</td><td>0.554</td><td>0.688</td><td>0.094</td></tr><tr><td>QCNN†</td><td>0.517</td><td></td><td></td><td></td></tr><tr><td></td><td>0.526 0.528</td><td>0.500</td><td>0.561 0.581</td><td>0.054</td></tr><tr><td>QPG (quantum edges)† QSVM-Pegasos†</td><td>0.523</td><td>0.550 0.508</td><td>0.557 0.573</td><td>0.059 0.074</td></tr><tr><td>QSVM-fixed†</td><td>0.522 0.513</td><td>0.547</td><td>0.577</td><td>0.060</td></tr><tr><td>TQEK†</td><td>0.519 0.454</td><td></td><td>0.665</td><td>0.071</td></tr><tr><td>VQC†</td><td>0.505 0.469</td><td>0.570 0.540</td><td>0.599</td><td>0.029</td></tr><tr><td>CNN (1-D)</td><td>0.540 0.514</td><td>0.568</td><td>0.619</td><td>0.102</td></tr><tr><td>MLP</td><td>0.523 0.496</td><td>0.553</td><td>0.599</td><td>0.059</td></tr><tr><td>MLP (no graph)</td><td>0.608 0.632</td><td>0.652</td><td>0.578</td><td>0.220</td></tr><tr><td>PG (RBF edges)</td><td>0.573 0.612</td><td></td><td>0.494</td><td>0.139</td></tr><tr><td>PG (correlation edges)</td><td>0.585 0.610</td><td>0.601</td><td>0.545</td><td>0.158</td></tr><tr><td>SVM (RBF), 8 feat.</td><td></td><td>0.602</td><td></td><td></td></tr><tr><td></td><td>0.526 0.499</td><td>0.561</td><td>0.603</td><td>0.076</td></tr><tr><td>SVM (RBF), 2,000 feat.</td><td>0.590</td><td>0.604 0.651</td><td>0.567</td><td>0.182</td></tr><tr><td>SVM (Linear), 2,000 feat.</td><td>0.576</td><td>0.593 0.594</td><td>0.562</td><td>0.152</td></tr><tr><td>Trainable-RBF UCLA-CNP</td><td>0.571</td><td>0.682 0.566</td><td>0.238</td><td>0.104</td></tr><tr><td colspan="5"></td></tr><tr><td>QCNN†</td><td>0.585</td><td>0.610</td><td>0.589 0.566</td><td></td><td>0.168</td></tr><tr><td>QSVM-Pegasos†</td><td>0.561</td><td>0.652</td><td></td><td>0.324</td><td>0.094</td></tr><tr><td>QSVM-fixed†</td><td>0.554</td><td>0.584</td><td>0.579 0.519</td><td></td><td>0.101</td></tr><tr><td>TQEK†</td><td>0.558</td><td>0.594</td><td>0.569</td><td>0.507</td><td>0.111</td></tr><tr><td>VQC†</td><td>0.486</td><td>0.509</td><td>0.528</td><td>0.485</td><td>-0.030</td></tr><tr><td>CNN (1-D)</td><td>0.544</td><td>0.569</td><td>0.555</td><td>0.518</td><td>0.093</td></tr><tr><td>MLP</td><td>0.597</td><td>0.630</td><td>0.611</td><td>0.539</td><td>0.197</td></tr><tr><td>SVM (RBF), 8 feat.</td><td>0.540</td><td>0.570 0.569</td><td>0.506</td><td></td><td>0.077</td></tr><tr><td>SVM (RBF), 2,000 feat.</td><td>0.567</td><td>0.608 0.591</td><td>0.508</td><td></td><td>0.120</td></tr><tr><td></td><td>0.557</td><td></td><td>0.629</td><td></td><td></td></tr><tr><td>SVM (Linear), 2,000 feat.</td><td></td><td>0.627</td><td></td><td>0.408</td><td>0.083</td></tr><tr><td>Trainable-RBF</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.602</td><td>0.634</td><td>0.610 0.557</td><td></td><td>0.201</td></tr></table>

† denotes a quantum model.

The variational classifier is a different matter. H-VQC did not improve, and its deficit is the strongest effect in the study. Across four cohorts, nine of forty pairwise comparisons reach $p < 0 . 0 5$ against an expectation of two, three survive Bonferroni correction, and eight of the nine have H-VQC on the losing side. On REST-meta-MDD it loses to H-Linear by 0.049 $( p = 0 . 0 0 4 )$ and to H-MLP by 0.048 $( p = 0 . 0 4 8 )$ . The evaluated configuration, a ZZ feature map with a RealAmplitudes ansatz and a parity read-out, performed worse than the corresponding classical heads, whereas the convolutional quantum head with a Pauli-Z read-out did not. H-Linear reaches within 0.02 of the best classical head on every cohort, so the projection carries most of the signal; additional read-out ablations are required to determine which component of the variational configuration causes the deficit.

## B. Ensembles Over Disjoint Feature Blocks

Where the projection compresses all features into the register, ensembling sidesteps the cap: the top 32 features are split into four disjoint blocks, one narrow model is trained per block, and the outputs are combined with learned weights. Every circuit stays at eight qubits while the ensemble sees 32 features. Two comparators separate the two things that could produce a gain. A classical ensemble over the identical blocks with the identical combiner isolates the model; a single classical model given all 32 features at once bounds what the partition itself costs.

Ensembling helps the classical arm and the quantum arm equally. On ABIDE I the block ensemble reaches 0.618 for the quantum members against 0.622 for the matched classical ensemble, and on REST-meta-MDD 0.540 against 0.542; of six pairwise comparisons across these two cohorts, none reaches $p < 0 . 0 5$ (smallest $p = 0 . 1 7 8 )$ . On ABIDE I, both ensembles exceed the single classical model given all 32 features at once (0.603), so the partition is worth roughly +0.017 whether the members are quantum or classical, but on REST-meta-MDD the single model leads (0.564), so even the ensembling benefit does not replicate.

## C. Data Re-uploading

The underfitting observed in Section VIII-B points to capacity rather than width. Data re-uploading [41] re-injects the features between convolution stages, each time with its own trainable gain, raising the order of Fourier terms the circuit can express, a capacity that a single input encoding cannot reach, at an unchanged register width. We compare a plain QCNN against re-upload depths one and two, alongside the matched classical heads C-CNN, C-MLP and C-Linear, and record training accuracy as well as test accuracy, because the claim being tested is specifically about capacity.

Re-uploading does not improve on the plain QCNN on any cohort. The median per-fold change at depth one is 0.000 on

ABIDE I, ADHD-200 and REST-meta-MDD $( p \geq 0 . 6 3 )$ and −0.043 on $\mathrm { U C L A - C N P } \left( p = 0 . 1 1 \right)$ , and depth two is no better $( p \geq 0 . 0 9$ throughout). Across all four cohorts, 60 pairwise comparisons among the six models yield nine results at $p <$ 0.05 uncorrected against an expectation of three by chance, and none survives Bonferroni correction $( 0 . 0 5 / 6 0 = 0 . 0 0 0 8 3 ;$ the smallest p anywhere is 0.008).

(a)

All nine sit on the two smallest cohorts, and seven of them have the same losing side: C-CNN. On ADHD-200, all three quantum models beat it (medians +0.038 to +0.048, $p \ = \ 0 . 0 1 6 \ { \mathrm { ~ t o ~ } } \ 0 . 0 3 1 )$ , but so does C-Linear $( + 0 . 0 6 2 , ~ p ~ =$ 0.031); on UCLA-CNP the plain QCNN beats it $( + 0 . 0 8 7 ,$ $p \ = \ 0 . 0 0 8 )$ , and so do C-MLP (+0.087, $p \ = \ 0 . 0 1 4 )$ and C-Linear (+0.111, $p = 0 . 0 1 4 )$ , at the same effect size. The remaining two go against the quantum arm, with C-Linear beating both re-uploading circuits on UCLA-CNP (+0.045 and $+ 0 . 0 4 4 , p = 0 . 0 3 1 )$ . The 1-D CNN fails to train at these sample sizes, sitting at or below chance (0.482 on ADHD-200, 0.500 on UCLA-CNP) while reaching 0.599 on ABIDE I.

## D. Capacity Converts to Overfitting, Not Accuracy

Recording training accuracy clarifies this null result: across the evaluated cohorts, models with higher training accuracy did not achieve higher held-out accuracy. Across the six reuploading models, mean training accuracy predicts the overfit gap with a Pearson correlation of $r = + 0 . 9 8 3 \ ( p = 0 . 0 0 0 4 )$ on REST-meta-MDD, $+ 0 . 9 4 9 \ : \ : ( p \ : = \ : 0 . 0 0 4 )$ on ABIDE I, +0.731 $( p = 0 . 0 9 9 )$ on ADHD-200 and +0.612 $( p = 0 . 1 9 7 )$ on UCLA-CNP, positive on all four and significant on the two best-powered cohorts. Parameter count tracks the gap even more consistently (Spearman $\rho = + 0 . 9 4 , + 0 . 9 4 , + 0 . 8 9$ +1.00; significant on all four). Training accuracy is meanwhile negatively related to test accuracy on three cohorts.

Re-uploading did what its theory claims: it raised training accuracy from 0.651 to 0.655 on ABIDE I and from 0.575 to 0.578 on REST-meta-MDD, so the added Fourier capacity is real and usable; it did not, however, improve held-out accuracy. On ABIDE I the best re-uploading model is an 18-parameter linear layer, with the best accuracy, the best AUC, the smallest gap and a runtime two orders of magnitude below the QCNN. On REST-meta-MDD, all six models sit inside a 0.008 band.

This revises the reading of Section VIII-B: additional capacity did not raise the QCNN’s held-out accuracy. Increasing capacity, by widening the register, learning a projection, ensembling, or re-uploading, improved training fit but did not improve held-out performance in the evaluated configurations, whether the mechanism adding it was quantum or classical.

Taken with Sections VII and VI, six architectural families have now been tested: fidelity kernels, population graphs, the Qiskit model suite, learned projections, block ensembles and data re-uploading. In every one, each intervention that helps the quantum arm helps its matched classical twin at least as much.

## X. DISCUSSION: WHEN DOES QUANTUM HELP?

## A. Where the Arms Differ

Comparing training and test accuracy reveals one aspect in which the model families differ. A QCNN reads out a single

(b)  
![](images/dd1c6ce34f8756ca7ab32f5f4c4c22645074fcb9e402ca4fabc2a7211f47a6b1.jpg)

![](images/3fcdfe4d0699cd6c8afaaa260ac30ead25705909063ac330cb9dac5067aaa4e6.jpg)  
Figure 8. Overfitting gap by cohort, and test accuracy and overfitting gap against training-set size on ABIDE I.

Pauli-Z expectation bounded in $[ - 1 , 1 ]$ , and the affine map to logits adds two parameters. All information the circuit learns is therefore conveyed through this bounded scalar output. This bottleneck may contribute to the smaller train–test gap observed in the quantum models, but we have not isolated it; the control would be a classical network of matched size ending in the same bounded scalar.

The overfitting gap, training minus test accuracy, of the quantum arms averaged over the three QCNN variants of Section IX-C is smaller than that of both neural comparators on all four cohorts (Fig. 8(a)), though the linear classifier, the smallest model, overfits least on every cohort, and across all six models the gap tracks parameter count (Section IX-D). The difference is therefore consistent with lower effective capacity; it does not isolate a quantum-specific mechanism and does not, by itself, establish a quantum advantage. The margin over the MLP is 0.034, 0.043, 0.043 and 0.051 for cohorts of 2,428, 1,035, 619 and 226 subjects (paired Wilcoxon $p ~ < ~ 0 . 0 0 1$ $p = 0 . 0 0 1 , 0 . 0 4 7$ and 0.193). These observations should be interpreted descriptively rather than inferentially: with four cohorts the exact permutation null gives $p = 0 . 3 3 3$ for a rank correlation of this size, and the ratio of the two gaps orders the cohorts the other way (1.72, 2.13, 1.27, 1.23).

## B. Analysis of Positive Quantum–Classical Differences

We examined the individual matched comparisons to identify patterns not reflected in the aggregate results. Across all runs, we make 284 matched comparisons, covering every quantum/classical pair, on accuracy, F1, AUC, and specificity. The quantum arm is ahead in 115 of them (40.5%) and significantly ahead in 30 (10.6%; Table VI). There, “ahead” counts a positive median difference for the quantum arm, “sig.” counts $p < 0 . 0 5$ on a paired Wilcoxon test, and “clean”

Table VI  
COMPLETE SWEEP OF MATCHED QUANTUM–CLASSICAL COMPARISONS BY METRIC.
<table><tr><td>Metric</td><td>Pairs</td><td>Ahead</td><td>Sig.</td><td>Clean</td></tr><tr><td>Accuracy</td><td>72</td><td>23</td><td>5</td><td>1</td></tr><tr><td>F1</td><td>72</td><td>35</td><td>4</td><td>2</td></tr><tr><td>AUC</td><td>68</td><td>34</td><td>16</td><td>6</td></tr><tr><td>Specificity</td><td>72</td><td>23</td><td>5</td><td>3</td></tr><tr><td>Total</td><td>284</td><td>115</td><td>30</td><td>12</td></tr></table>

![](images/c42e706bf30586e44c0bc7152aa3ad8b4995d38b5611358be6bb4d5a6ec76dea.jpg)  
Figure 9. Accuracy against Matthews correlation for every model on every cohort, pooled over folds.

excludes wins in which either arm is not classifying (sensitivity or specificity below 0.30, or $| \mathrm { M C C } | < 0 . 0 2$ , on the confusion matrix pooled over folds) or the comparator fails to train (C-CNN).

Eighteen of the thirty are screened out (Fig. 9). Plotting accuracy against Matthews correlation for every model on every cohort, pooled over folds, separates the models that classify from those that do not: models predicting close to a single class sit far below the trend, scoring near the majorityclass rate on accuracy and near zero on MCC, and the annotated case, QSVM-Pegasos on ADHD-200, returns zero true positives. Screening the classical arms on these pooled matrices flags four that are not classifying: C-CNN on UCLA-CNP (MCC = −0.005), the suite’s 1-D CNN (−0.015) and C-MLP (+0.015) on ADHD-200, and Trainable-RBF on REST-meta-MDD, which is near-all-positive (specificity 0.276). Fifteen significant wins are against C-CNN; one is a QSVM-Pegasos specificity win earned by predicting a single class; one is TQEK’s +0.471 specificity against the degenerate Trainable-RBF; and one is the QCNN’s ADHD-200 AUC win against the degenerate 1-D CNN. These cases do not provide strong evidence regarding quantum-specific behaviour.

Twelve survive, and half of them are on AUC, a ranking metric.

• QCNN against its matched classical CNN on ABIDE I. AUC +0.037 $( p ~ = ~ 0 . 0 3 3 )$ , specificity +0.038 $( p \_ ) =$ 0.030), and accuracy +0.044 at $\begin{array} { r l r } { p } & { { } = } & { 0 . 0 5 8 . } \end{array}$ The threshold-independent AUC lead extends to ADHD-200 (+0.059, $p = 0 . 0 4 7 )$ , reported separately as that comparator does not separate the classes.

• TQEK against a trainable classical kernel on ABIDE I. AUC +0.064 $( p = 0 . 0 0 1 2 )$ , specificity +0.111 $( p = 0 . 0 0 0 6 )$ , accuracy +0.028 $( p = 0 . 0 4 9 )$ . This represents the largest observed single-cohort effect, against a comparator that is functional in this cohort. However, the effect was not reproduced in the remaining cohorts (−0.013 on REST-meta-MDD, −0.045 on UCLA-CNP).

• Re-uploading against C-MLP on ABIDE I. AUC +0.055 (p = 0.0094), +0.043 $( p = 0 . 0 1 9 )$ and +0.053 $( p = 0 . 0 0 3 2 )$ at depths zero, one and two.

• Learned-projection quantum head against its linear control. AUC +0.017 $( p = 0 . 0 2 4 )$ on ABIDE I.

Matched versus strongest classical comparators. These results refer to matched comparators rather than to the highestperforming classical model: each quantum model is set against the classical model it corresponds to, which we regard as the fairest test of the architecture. Under this comparison, the QCNN achieved higher AUC than its matched 1-D CNN on ABIDE I, although this difference does not establish superiority over the strongest classical models. Against the best classical AUC in each run, the quantum arm ties on ABIDE I (0.679 against 0.679 for the linear classifier) and trails on the other three cohorts (0.548 against 0.563; 0.566 against 0.582; 0.590 against 0.603), so, in general, the quantum models did not outperform the strongest classical alternatives.

Why AUC and not accuracy. The quantum arm leads on AUC in 34 of 68 comparisons but on accuracy in only 23 of 72. AUC measures ranking; accuracy measures ranking and where the decision threshold falls, and a bounded Pauli-Z expectation mapped to logits by a two-parameter affine layer is a potentially limited calibration mechanism, suggesting threshold calibration as a possible explanation. We tested it on all four cohorts, choosing the threshold that maximises Youden’s J (sensitivity plus specificity minus one) on the training scores of each fold and applying it unchanged to the test scores, identically for every arm. No arm gains: averaged over the cohorts, calibration changes accuracy by −0.000 for the quantum arm, +0.009 for C-CNN, +0.002 for C-MLP and −0.009 for C-Linear, and by −0.001 on ABIDE I, where the quantum arm has its highest AUC (0.679). Training-set optimisation of Youden’s J therefore did not explain the AUCto-accuracy discrepancy: the quantum arm leads C-CNN on both ranking and thresholding, and is on par with the linear classifier on both.

## C. Assessing Evidence for Quantum Advantage

Three observations limit the evidence for a quantum advantage. The quantum arm is ahead in only 115 of 284 matched comparisons (40.5%), where a model with no systematic advantage would be ahead in roughly half. Thirty significant results exceed the fourteen expected by chance, but the four metrics are computed on identical folds and are strongly correlated, so the excess cannot be read as sixteen real effects, and none survives Bonferroni correction across the 284 comparisons. Finally, the leads do not replicate: almost every clean win is on ABIDE I, the cohort exhibiting the strongest overall classification performance (best classical accuracy exceeds the majority-class rate by 0.104, against 0.036 for REST-meta-MDD and 0.010 for ADHD-200), and TQEK’s result, the strongest in the study, reverses on both other cohorts where it can be measured.

![](images/8b06ac04fff4513087aca2a76e81ffea2990c7908781b1aed5f62c801358b829.jpg)  
Figure 10. Quantum-minus-classical accuracy for the three central matched pairs, per cohort (site folds in brackets) and combined across cohorts by a random-effects model.

Non-significance is not equivalence, so Fig. 10 bounds the three central matched differences. Combined across cohorts with a random-effects model [42], the quantum kernel differs from its matched RBF kernel by +0.002 in accuracy (95% CI −0.009 to +0.014), the QCNN from its CNN by +0.008 (−0.025 to +0.041), and the hybrid QCNN from its hybrid CNN by −0.008 (−0.021 to +0.004). Under the fitted random-effects model, the 95% confidence intervals are therefore inconsistent with average quantum improvements larger than 1.4, 4.1 and 0.4 percentage points for the three respective comparisons. ADHD-200, with seven folds, constrains little, so these bounds rest mainly on ABIDE I and REST-meta-MDD.

A more conservative conclusion is that, on the cohort with the strongest overall classification performance, a quantum convolutional network ranks subjects at least as well as its matched classical counterpart, on a threshold-free metric, with significance. This finding supports competitiveness rather than a reproducible advantage, and it rests on one cohort.

## D. The Low-Data Regime, Tested Directly

Section X-A left a falsifiable prediction: if the quantum arm’s smaller overfitting gap widens as data become scarce, there should be a training size below which it converts into an accuracy advantage. We tested it on ABIDE I by holding the leave-site-out folds and test sets fixed and subsampling only the training pool to m subjects, three independent draws per fold and size.

Fig. 8(b) shows that the gap separates the arms at every size while accuracy does not.

The smaller gap holds at every size. The quantum arm has a smaller overfitting gap than the matched MLP at all four training sizes, without exception, falling from 0.248 at $m = 1 0 0$ to 0.160, 0.099 and 0.046 at 200, 400 and 800, against 0.363, 0.253, 0.159 and 0.094: the MLP overfits 1.46× as much at m = 100, 1.58× at 200, 1.61× at 400 and 2.02× at 800. The ratio grows with sample size rather than shrinking with it: the quantum model maintains a consistently smaller train–test gap, but this relative difference does not increase in the smallest-sample conditions.

The direction of the accuracy effect is the opposite of the prediction. The quantum arm is the best of the three at m = 400 (0.586) and m = 800 (0.613), and not at the two smallest sizes, where the linear classifier leads (0.550 at m = 100, 0.596 at $m = 2 0 0 )$ . Only one comparison reaches significance, and it favours quantum: +0.010 over the MLP at m = 400 $( p = 0 . 0 4 0 )$ . A low-data quantum advantage would have produced the reverse ordering.

## E. Sources of Apparent Quantum Advantage

Four mechanisms can produce an apparent quantum advantage, and the matched controls of this study are designed to remove them. Each is common in the published literature, and evaluation defects of this kind have been documented in previous studies: a recent audit of a hybrid quantum– classical skin-lesion classifier found 22.8% of lesions on both sides of the train–test split, inflating every previously reported metric [43].

An unmatched feature budget. Comparing an eightqubit model against a classical model given 2,000 features measures dimensionality, not the kernel; an unmatched configuration shows a quantum deficit for exactly this reason, and it disappears under matching (Section V-B). The budget itself is costly: on REST-meta-MDD, compressing 19,900 pairwise correlations to eight components costs more than any quantum–classical difference measured in this study, while widening the register to admit more features increases the concentration of pairwise fidelity (Section VIII-A).

Reliance on a single cohort. Positive results on one cohort did not replicate on the others anywhere in this study: TQEK gains +0.028 on ABIDE I but loses on REST-meta-MDD and UCLA-CNP (Section VI), and intervention results on six-fold subsets did not carry over to the full cohorts.

A comparator that fails to train. On ADHD-200, all three quantum models beat the classical CNN at $p \leq 0 . 0 3 1$ , which may initially be interpreted as a quantum advantage until one observes that C-Linear beats it by a larger margin: the 1-D CNN sits at chance on the small cohorts (Section IX-C), and a single classical comparator would have made this a positive finding.

Unreported hyperparameter selection. The fidelity kernel spans effective dimensions from near-constant to near-identity as bandwidth varies (Section V-D); a bandwidth chosen post hoc can place it almost anywhere, and its default leaves it memorising the training set.

These are the reasons every comparison in this paper carries more than one classical arm, reports all cohorts rather than the favourable ones, and states both uncorrected and corrected significance.

## F. Comparison with Published Results

The proposed pipeline achieves performance that is competitive in some settings but below state-of-the-art results in others: competitive on ABIDE I, below the strongest published results on ADHD-200 and REST-meta-MDD (Table VII). This outcome reflects the design goals of the study. The classical baseline is deliberately simple: supervised selection plus a standard classifier with no multimodal fusion, no learned harmonisation, and no architecture search, because its role is to give the quantum comparison a well-tuned reference rather than to maximise benchmark performance. On ABIDE I, the protocol-matched anchor is Abraham et al., whose pipeline reaches about 0.67 including prediction on unseen sites [44]; our linear SVM on the full correlation vector reaches 0.671, matching that level, while a population GCN reaches 0.70 under ten-fold cross-validation [45], which evaluates a different generalisation setting from leave-site-out validation. On ADHD-200, a model given only phenotypic data achieved the highest accuracy of the 21 teams in the competition (0.625, on a three-class task) [46], which suggests that phenotypic information may contribute substantially to performance on this benchmark. On REST-meta-MDD, an ensemble graph neural network reaches 0.73 under leave-siteout validation [47], above our 0.608. On UCLA-CNP, a deep graph convolutional network reaches 0.82 on schizophrenia versus controls under ten-fold cross-validation [48], against our 0.602, but on a different task and 172 subjects, so it is a reference point rather than a like-for-like comparison.

Table VII  
THIS WORK AGAINST PUBLISHED RESULTS ON THE SAME COHORTS, WITH THE VALIDATION PROTOCOL OF EACH.
<table><tr><td rowspan=1 colspan=1>Cohort</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Protocol</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>ABIDE I</td><td rowspan=1 colspan=1>Abraham et al. [44]Population GCN [45]This work (linear SVM,full correlation)</td><td rowspan=1 colspan=1>LSO10-foldLSO</td><td rowspan=1 colspan=1>~0.670.700.671</td></tr><tr><td rowspan=1 colspan=1>ADHD-200</td><td rowspan=1 colspan=1>Phenotypic data alone,three-class [46]This work (SVM-RBF)</td><td rowspan=1 colspan=1>held-outLSO</td><td rowspan=1 colspan=1>0.6250.571</td></tr><tr><td rowspan=1 colspan=1>REST-meta-MDD</td><td rowspan=1 colspan=1>Ensemble GNN [47]This work (MLP, nograph)</td><td rowspan=1 colspan=1>LSOLSO</td><td rowspan=1 colspan=1>0.730.608</td></tr><tr><td rowspan=1 colspan=1>UCLA-CNP</td><td rowspan=1 colspan=1>Deep graph CNN,schizophrenia vs.controls [48]This work(Trainable-RBF)</td><td rowspan=1 colspan=1>10-fold10-fold</td><td rowspan=1 colspan=1>0.820.602</td></tr></table>

Reported accuracies on ABIDE I span from below 0.70 under leave-site-out validation to above 0.90 under random splits, and the spread is protocol rather than method. Three conditions must match before two numbers can be placed side by side: the validation protocol, since random k-fold splits let a model exploit site-specific artefacts while leave-site-out forbids it; the modality and covariates, since a model given phenotypic variables is solving a different problem; and the parcellation, since atlas choice changes both node count and signal.

DAGMNet [9] illustrates all three at once and is the highest ABIDE I accuracy we encountered (0.9159, AUC 0.9680). It is ten-fold rather than leave-site-out; it fuses sMRI, fMRI, and phenotype rather than using connectivity alone; it reports on the Harvard-Oxford, Eickhoff-Zilles (EZ) and AAL atlases rather than CC200; and its phenotypic inputs include study location, so site identity is available to the model while subjects from the same site appear in both training and test folds, and may therefore contribute to predictive performance under the reported validation protocol.

## XI. LIMITATIONS

Network-level interpretation. The models operate on whole-brain connectivity without regional priors, and several perform close to chance on some cohorts; Table I and Fig. 1 therefore summarise previously reported findings rather than results of this study. Attention weights are available but are not readily interpretable when classification performance is close to chance.

Correlation-derived inputs. Every classifier in this study, quantum or classical, receives features derived from Pearson correlation: selected entries of the correlation vector, its principal components, or regional connectivity profiles. Whether quantum similarity captures non-linear dependence between raw regional BOLD signals, which motivates this work, was therefore not tested; doing so would require encoding the time series directly.

Simulation and register width. The main results come from exact statevector simulation; the kernel result holds under simulated shot and depolarising noise (Section V-C), while gate-level noise and decoherence on hardware remain untested. In the evaluated datasets, the usable fidelity range narrowed as register width increased (Section VIII-A), while feature capacity demands the opposite, and although fidelity concentration need not limit variational models (Section VIII-B), simulation cost does. Whether the trend holds on hardware that does not pay that cost remains open.

Statistical scope. The reported intervals are computed across folds rather than across initialisations. Repeating the ABIDE I suite at four seeds indicates the size of the omitted variation: the largest standard deviation of mean accuracy across seeds is 0.017 (QSVM-Pegasos), and the two kernel models without a stochastic component are identical at every seed. Differences smaller than 0.017 between stochastic models should therefore be interpreted cautiously; the matched differences reported here nevertheless keep their sign across seeds, with the single exception of VQC against its matched MLP. The permutation null covers the kernel arm on ABIDE I only; the neural and graph arms were not permuted. For the graph pipeline, the test is less informative, because every fold of both fidelity-graph models predicts a single class.

Graph construction and baselines. The adjacency is directed and unsymmetrised, and k is fixed at 20, so the effect of either choice on node distinctiveness and accuracy was not measured. The evaluated baselines were intentionally selected to support matched comparisons rather than to represent the latest adaptive, dynamic or hypergraph learning architectures; re-implementing such a model under leave-siteout folds would position these results relative to the state of the art.

## XII. CONCLUSIONS

This study benchmarked near-term quantum models against matched classical baselines for functional-connectivity classification on four cohorts, three of them under leave-siteout validation. Most quantum models matched their classical counterparts, but none consistently outperformed them, so the results do not demonstrate a reproducible quantum advantage across the evaluated cohorts.

For fidelity-based graph construction, each added qubit leaves fewer region pairs above every fidelity floor tested, reflecting kernel concentration [36] rather than an implementation fault, and any trainable layer applied after the encoding cancels in the fidelity, so trainable similarity must be placed inside or between encodings, or in classical layers after the overlap (Sections VIII-A and VIII-C). At eight qubits, the quantum kernel and a classical RBF given the same eight features are statistically indistinguishable on all four cohorts across 62 folds (Section V-B), and under a random-effects model the 95% confidence interval excludes an average advantage above 1.4 percentage points (Fig. 10). Four attempts to lift the resulting feature cap, widening the register, learning a projection, ensembling over disjoint blocks, and adding data re-uploading, each helped the matched classical twin at least as much as the quantum arm in the evaluated experiments. Only the parity-read-out architecture consistently underperformed its matched classical counterpart.

One of the most broadly applicable findings concerns model capacity. Models with higher training accuracy did not achieve higher held-out accuracy in the evaluated comparisons: training fit tracks the overfitting gap closely on both well-powered cohorts $( r ~ = ~ + 0 . 9 5$ on ABIDE I, $r ~ = ~ + 0 . 9 8$ on RESTmeta-MDD) while relating negatively to held-out accuracy, and capacity added by any mechanism, quantum or classical, raised training fit without raising held-out accuracy. Across 284 matched comparisons, the quantum arm leads significantly in 30; 18 of these involve a predictor that is not classifying or fails to train, and half of the remaining 12 are on AUC, a ranking metric, with none surviving Bonferroni correction across all 284 comparisons.

These findings are specific to classically simulated circuits, fidelity-based topology and the model families tested here. The matched evaluation protocol and released per-fold results provide a reproducible benchmark for future quantum and classical studies.

## ACKNOWLEDGMENT

The authors thank the ABIDE, ADHD-200, REST-meta-MDD and UCLA-CNP consortia for collecting and sharing the data analysed here. Fig. 1 was rendered with BrainPainter [49].

## DATA AVAILABILITY AND ETHICS

All cohorts were obtained under each consortium’s access terms; ethical approval and informed consent were obtained by the contributing sites, and no new data were collected. Code, per-fold results, scripts and fold definitions are available at https://github.com/r11up/qml-fmri-baseline.

[1] U. Poudel, S. Jakhar, P. Mohan, and A. Nepal, “AI in mental health: A review of technological advancements and ethical issues in psychiatry,” Issues in Mental Health Nursing, vol. 46, no. 7, pp. 693–701, 2025, doi: 10.1080/01612840.2025.2502943.

[2] B. A. Vachha, V. A. Kumar, J. J. Pillai, J. S. Shimony, J. Tanabe, and H. I. Sair, “Resting-state functional MRI: Current state, controversies, limitations, and future directions—AJR expert panel narrative review,” American Journal of Roentgenology, vol. 225, no. 3, Art. no. e2432163, 2025, doi: 10.2214/AJR.24.32163.

[3] S. Vidya, K. Gupta, A. Aly, A. Wills, E. Ifeachor, and R. Shankar, “Identification of critical brain regions for autism diagnosis from fMRI data using explainable AI: An observational analysis of the ABIDE dataset,” eClinicalMedicine, vol. 88, Art. no. 103452, 2025, doi: 10.1016/j.eclinm.2025.103452.

[4] S. M. Motlaghian et al., “Nonlinear functional network connectivity in resting functional magnetic resonance imaging data,” Human Brain Mapping, vol. 43, no. 15, pp. 4556–4566, 2022, doi: 10.1002/hbm.25972.

[5] Y. Jin and X. Zhu, “Oversmoothing alleviation in graph neural networks: A survey and unified view,” Knowledge and Information Systems, vol. 67, no. 12, pp. 11259–11285, 2025, doi: 10.1007/s10115-025- 02548-6.

[6] M. Schuld and N. Killoran, “Quantum machine learning in feature Hilbert spaces,” Physical Review Letters, vol. 122, no. 4, Art. no. 040504, 2019, doi: 10.1103/PhysRevLett.122.040504.

[7] U. Poudel and T. P. Jose, “Influence of artificial intelligence on the future of psychiatry: Insights from recent advancements,” Psychiatry, vol. 87, no. 4, pp. 372–373, 2024, doi: 10.1080/00332747.2024.2360373.

[8] F. Noman, R. C.-W. Phan, H. Ombao, and C.-M. Ting, “Adaptive graph learning with multi-graph convolutions for brain disorder classification,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2025 (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2025, pp. 56–65, doi: 10.1007/978-3-032-05162-2 6.

[9] L. Wang, X. Li, J. Yuan, and Y. Chen, “DAGMNet: Dual-branch attention-pruned graph neural network for multimodal sMRI and fMRI fusion in autism prediction,” Biomedicines, vol. 13, no. 9, Art. no. 2168, 2025, doi: 10.3390/biomedicines13092168.

[10] U. Poudel, J. Aryal, and S. Vairavasundaram, “Neuro-geospatial modelling of EEG affective states using literature-informed environmental context,” 2026, arXiv:2608.20807.

[11] N. A. Orka, M. A. Awal, P. Lio, G. Pogrebna, A. G. Ross, and\` M. A. Moni, “Quantum deep learning in neuroinformatics: A systematic review,” Artificial Intelligence Review, vol. 58, no. 5, Art. no. 134, 2025, doi: 10.1007/s10462-025-11136-7.

[12] P. Lamichhane and D. B. Rawat, “Quantum machine learning: Recent advances, challenges, and perspectives,” IEEE Access, vol. 13, pp. 94057–94105, 2025, doi: 10.1109/ACCESS.2025.3573244.

[13] J. Choi et al., “Early-stage detection of cognitive impairment by hybrid quantum-classical algorithm using resting-state functional MRI timeseries,” Knowledge-Based Systems, vol. 310, Art. no. 112922, 2025, doi: 10.1016/j.knosys.2024.112922.

[14] J. J. Park et al., “Resting-state fMRI analysis using quantum timeseries transformer,” in 2025 IEEE International Conference on Quantum Computing and Engineering (QCE), Albuquerque, NM, USA, 2025, pp. 2352–2363, doi: 10.1109/QCE65121.2025.00256.

[15] S. Saranya and R. Menaka, “A quantum-based machine learning approach for autism detection using common spatial patterns of EEG signals,” IEEE Access, vol. 13, pp. 15739–15750, 2025, doi: 10.1109/AC-CESS.2025.3531979.

[16] Y. Bagoun, A. Zinedine, and I. Berrada, “Autism spectrum disorder detection with quantum machine learning methods,” in Proc. 6th Int. Conf. Intelligent Computing in Data Sciences (ICDS), Marrakech, Morocco, 2024, pp. 1–6, doi: 10.1109/ICDS62089.2024.10756377.

[17] S. M. Turjya and M. Fawakherji, “Bridging deep learning and quantum computing for EEG-driven P300 targeted ASD response detection,” IEEE Access, vol. 13, pp. 214533–214561, 2025, doi: 10.1109/AC-CESS.2025.3646589.

[18] N. Huang, X. Li, X. Yang, L. Xie, G. Yang, and L. Zhou, “QG-WRN: A quantum-enhanced graph convolutional wide residual network for ASD diagnosis via neuroimaging sensing technology,” Sensors, vol. 26, no. 13, Art. no. 3997, 2026, doi: 10.3390/s26133997.

[19] B. T. T. Yeo et al., “The organization of the human cerebral cortex estimated by intrinsic functional connectivity,” Journal of Neurophysiology, vol. 106, no. 3, pp. 1125–1165, 2011, doi: 10.1152/jn.00338.2011.

[20] C.-G. Yan et al., “Reduced default mode network functional connectivity in patients with recurrent major depressive disorder,” Proceedings of the National Academy of Sciences of the USA, vol. 116, no. 18, pp. 9078– 9083, 2019, doi: 10.1073/pnas.1900390116.

[21] V. Havl´ıcek ˇ et al., “Supervised learning with quantum-enhanced feature spaces,” Nature, vol. 567, no. 7747, pp. 209–212, 2019, doi: 10.1038/s41586-019-0980-2.

[22] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and\` Y. Bengio, “Graph attention networks,” in Proc. 6th Int. Conf. Learning Representations (ICLR), Vancouver, BC, Canada, 2018.

[23] A. Di Martino et al., “The autism brain imaging data exchange: Towards a large-scale evaluation of the intrinsic brain architecture in autism,” Molecular Psychiatry, vol. 19, no. 6, pp. 659–667, 2014, doi: 10.1038/mp.2013.78.

[24] C. Craddock et al., “The Neuro Bureau Preprocessing Initiative: Open sharing of preprocessed neuroimaging data and derivatives,” Frontiers in Neuroinformatics, Conference Abstract: Neuroinformatics 2013, Stockholm, Sweden, 2013, doi: 10.3389/conf.fninf.2013.09.00041.

[25] The ADHD-200 Consortium, “The ADHD-200 Consortium: A model to advance the translational potential of neuroimaging in clinical neuroscience,” Frontiers in Systems Neuroscience, vol. 6, Art. no. 62, 2012, doi: 10.3389/fnsys.2012.00062.

[26] R. A. Poldrack et al., “A phenome-wide examination of neural and cognitive function,” Scientific Data, vol. 3, Art. no. 160110, 2016, doi: 10.1038/sdata.2016.110.

[27] R. C. Craddock, G. A. James, P. E. Holtzheimer, X. P. Hu, and H. S. Mayberg, “A whole brain fMRI atlas generated via spatially constrained spectral clustering,” Human Brain Mapping, vol. 33, no. 8, pp. 1914– 1928, 2012, doi: 10.1002/hbm.21333.

[28] M. Rosenblatt, L. Tejavibulya, R. Jiang, S. Noble, and D. Scheinost, “Data leakage inflates prediction performance in connectome-based machine learning models,” Nature Communications, vol. 15, Art. no. 1829, 2024, doi: 10.1038/s41467-024-46150-w.

[29] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. 3rd Int. Conf. Learning Representations (ICLR), San Diego, CA, USA, 2015.

[30] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, “On kernel-target alignment,” in Advances in Neural Information Processing Systems 14. Cambridge, MA, USA: MIT Press, 2002, pp. 367–373.

[31] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.

[32] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. 5th Int. Conf. Learning Representations (ICLR), Toulon, France, 2017.

[33] I. Cong, S. Choi, and M. D. Lukin, “Quantum convolutional neural networks,” Nature Physics, vol. 15, no. 12, pp. 1273–1278, 2019, doi: 10.1038/s41567-019-0648-8.

[34] G. Gentinetta, A. Thomsen, D. Sutter, and S. Woerner, “The complexity of quantum support vector machines,” Quantum, vol. 8, Art. no. 1225, 2024, doi: 10.22331/q-2024-01-11-1225.

[35] T. Hubregtsen, D. Wierichs, E. Gil-Fuster, P.-J. H. S. Derks, P. K. Faehrmann, and J. J. Meyer, “Training quantum embedding kernels on near-term quantum computers,” Physical Review A, vol. 106, no. 4, Art. no. 042431, 2022, doi: 10.1103/PhysRevA.106.042431.

[36] S. Thanasilp, S. Wang, M. Cerezo, and Z. Holmes, “Exponential concentration in quantum kernel methods,” Nature Communications, vol. 15, Art. no. 5200, 2024, doi: 10.1038/s41467-024-49287-w.

[37] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, “Barren plateaus in quantum neural network training landscapes,” Nature Communications, vol. 9, Art. no. 4812, 2018, doi: 10.1038/s41467-018- 07090-4.

[38] M. Cerezo, A. Sone, T. Volkoff, L. Cincio, and P. J. Coles, “Cost function dependent barren plateaus in shallow parametrized quantum circuits,” Nature Communications, vol. 12, Art. no. 1791, 2021, doi: 10.1038/s41467-021-21728-w.

[39] E. Grant, L. Wossnig, M. Ostaszewski, and M. Benedetti, “An initialization strategy for addressing barren plateaus in parametrized quantum circuits,” Quantum, vol. 3, Art. no. 214, 2019, doi: 10.22331/q-2019- 12-09-214.

[40] A. Skolik, J. R. McClean, M. Mohseni, P. van der Smagt, and M. Leib, “Layerwise learning for quantum neural networks,” Quantum Machine Intelligence, vol. 3, no. 1, Art. no. 5, 2021, doi: 10.1007/s42484-020- 00036-4.

[41] A. Perez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre,´ “Data re-uploading for a universal quantum classifier,” Quantum, vol. 4, Art. no. 226, 2020, doi: 10.22331/q-2020-02-06-226.

[42] R. DerSimonian and N. Laird, “Meta-analysis in clinical trials,” Controlled Clinical Trials, vol. 7, no. 3, pp. 177–188, 1986, doi: 10.1016/0197-2456(86)90046-2.

[43] M. R. A. Jamaludin and C. S. Kim, “Evaluation integrity in hybrid quantum-classical skin lesion classification: A data-leakage audit and lesion-level correction on HAM10000,” IEEE Access, vol. 14, pp. 101415–101428, 2026, doi: 10.1109/ACCESS.2026.3707673.

[44] A. Abraham et al., “Deriving reproducible biomarkers from multi-site resting-state data: An autism-based example,” NeuroImage, vol. 147, pp. 736–745, 2017, doi: 10.1016/j.neuroimage.2016.10.045.

[45] S. Parisot et al., “Disease prediction using graph convolutional networks: Application to autism spectrum disorder and Alzheimer’s disease,” Medical Image Analysis, vol. 48, pp. 117–130, 2018, doi: 10.1016/j.media.2018.06.001.

[46] M. R. G. Brown et al., “ADHD-200 Global Competition: Diagnosing ADHD using personal characteristic data can outperform resting state fMRI measurements,” Frontiers in Systems Neuroscience, vol. 6, Art. no. 69, 2012, doi: 10.3389/fnsys.2012.00069.

[47] K. Zheng, H. Lu, H. Wang, D. Hu, B. Chen, and B. Li, “Functional connectivity-based classification and subtyping of major depression for precision mental health: An ensemble graph neural network approach,” PLOS Digital Health, vol. 5, no. 3, Art. no. e0001261, 2026, doi: 10.1371/journal.pdig.0001261.

[48] G. Sunil, S. Gowtham, A. Bose, S. Harish, and G. Srinivasa, “Graph neural network and machine learning analysis of functional neuroimaging for understanding schizophrenia,” BMC Neuroscience, vol. 25, no. 1, Art. no. 2, 2024, doi: 10.1186/s12868-023-00841-0.

[49] R. V. Marinescu, A. Eshaghi, D. C. Alexander, and P. Golland, “Brain-Painter: A software for the visualisation of brain structures, biomarkers and associated pathological processes,” in Multimodal Brain Image Analysis and Mathematical Foundations of Computational Anatomy (Lecture Notes in Computer Science, vol. 11846). Cham, Switzerland: Springer, 2019, pp. 112–120, doi: 10.1007/978-3-030-33226-6 13.