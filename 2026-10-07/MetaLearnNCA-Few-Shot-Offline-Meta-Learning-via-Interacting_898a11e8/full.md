# MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata

Etienne Guichard & Stefano Nichele Department of Computer Science and Communication {etienne.guichard,stefano.nichele}@hiof.no

## Abstract

Few-shot meta-learning traditionally formulates task adaptation either as analytical gradient descent through unrolled computational graphs or as metric-based distance comparisons over flattened 1D feature vectors, which either incur costly test-time backpropagation or discard native 2D spatial geometry. In this work, we propose METALEARNNCA, a decentralized framework that achieves few-shot adaptation through the dynamical interaction of coupled Neural Cellular Automata (NCAs) without computing analytical gradients during inference. METALEARNNCA decomposes task adaptation into an Active-NCA, which executes task inference conditioned on a continuous 2D spatial memory grid termed the spatial program, and a learned Meta-NCA, which acts as a decentralized cellular optimizer by diffusing spatial error residuals across local neighborhoods to dynamically update this program. METALEARN-NCA is competitive against canonical meta-learners in-distribution (96.12% on Omniglot) with Out-Of-Distribution transfer gains on MNIST, KMNIST, and Fashion-MNIST transfer across 10 independent testing seeds across 1-, 5-, and 10-shot regimes (e.g., surpassing Prototypical Networks by +10.54% on 10-shot MNIST and a +3.87% gain on 10-shot Fashion-MNIST over FOMAML). Our results establish that robust, gradient-free learning-to-learn can emerge from decentralized cellular dynamics on non-von Neumann substrates.

## 1 Introduction

Few-shot meta-learning seeks to equip computational systems with the ability to acquire novel concepts rapidly from minimal supervision [Hospedales et al., 2021, Vinyals et al., 2016]. In biological organisms, this learning-to-learn capacity emerges not from monolithic, centralized gradient updates across billions of parameters, but through coordinated, decentralized cellular processes operating under strict physical and energetic constraints [Levin, 2019]. Recently, Neural Cellular Automata (NCAs) [Mordvintsev et al., 2020] have emerged as a compelling paradigm of decentralized computing. Characterized by uniform local transition rules, translational invariance, and extreme parameter efficiency, NCAs have demonstrated remarkable prowess across morphogenesis [Mordvintsev et al., 2020], robust functional regeneration, texture synthesis [Niklasson et al., 2021], and spatial reasoning tasks—often utilizing orders of magnitude fewer parameters than conventional deep architectures.

Crucially, recent investigations reveal that the representational capacity of NCAs extends beyond static pattern formation. Studies have demonstrated that NCAs organize their latent, non-observable channels into structured functional representations [Guichard and Nichele, 2026, Montero et al., 2026, Randazzo et al., 2020]. For instance, NCAs can learn reusable functional primitives in hidden state channels that are subsequently recombined to execute hierarchical, multi-stage behaviors [Guichard et al., 2025b, Montero et al., 2026]. Furthermore, partitioning NCA states into private, immutable channels enables universal cel lular computation, where distinct sub-channels act as isolated memory registers or programmable routing mechanisms [Bena et al.´ , 2025]. In discriminative settings, NCAs have been shown to encode orthogonal, task-relevant representations within their latent channels; remarkably, features distilled from one NCA can bootstrap another to achieve competitive one-shot recognition [Guichard and Nichele, 2026]. This body of evidence points toward an unexplored frontier: the localized, recurrent hidden channels ofNCAs provide a natural substrate for decentralized, gradient-free meta-learning.

In contrast, current deep meta-learning paradigms suffer from fundamental trade-offs: gradient-based approaches [Finn et al., 2017, Nichol et al., 2018] require unrolling computational graphs and executing costly backpropagation during test-time inference, while black-box and in-context methods [Santoro et al., 2016, Brown et al., 2020] avoid test-time gradients only at the expense of massive parameter footprints $( 1 0 ^ { 8 } – 1 0 ^ { 1 1 }$ parameters), quadratic sequence complexity, and an absence of native 2D spatial inductive biases.

To bridge this divide, we propose MetaLearnNCA, a decentralized framework that achieves few-shot offline adaptation without computing gradients at test time. MetaLearnNCA decomposes task adaptation into the dynamical interaction of two localized cellular systems:

• An Active-NCA, which executes the primary task (e.g., visual character classification) conditioned on a localized, spatial memory grid termed the spatial program

• A Meta-NCA, which functions as a decentralized, learned cellular optimizer. Rather than calculating explicit gradients via chain-rule backpropagation, the Meta-NCA receives spatial residual error maps computed on the support set. Over a sequence of localized homogeneous transition functions, the Meta-NCA processes these residual errors and writes updated steering signals directly into the spatial program.

During adaptation, the Active-NCA and Meta-NCA communicate iteratively in a closed feedback loop: task inference produces residual errors, and cellular dynamics refines the spatial program. Once adaptation on the support set concludes, the resulting spatial program is frozen and broadcast across query instances. The query predictions are then computed purely through forward cellular unrolling—demanding zero gradient computation at adaptation or inference time.

Contributions. (1) We introduce METALEARNNCA, an offline meta-learning framework realizing fewshot adaptation via coupled NCAs that eliminates test-time backpropagation while maintaining a compact < 120k parameter footprint; (2) we formulate a spatially distributed conditioning mechanism where a learned Meta-NCA translates pixel-wise residual errors into persistent pre-activation program channels; and (3) we demonstrate good empirical performance,with competetive performance on in-distribution (96.12% on Omniglot) with Out-Of-Distribution transfer gains on MNIST, KMNIST, and Fashion-MNIST.

## 2 Related Work

Our work operates at the intersection of few-shot meta-learning, self-organizing Neural Cellular Automata, contrastive metric learning, and decentralized, biologically inspired credit assignment.

## 2.1 Few-Shot Meta-Learning and Test-Time Adaptation

Existing approaches in Few-shot Meta-Learning broadly divide into three paradigms: Gradient-Based Meta-Learning, Black-Box and Memory-Augmented Meta-Learning, and Metric-Based Learning.

Optimization-centric algorithms, such as FOMAML [Finn et al., 2017] and Reptile [Nichol et al., 2018], Meta-SGD [Li et al., 2017], and implicit formulations [Rajeswaran et al., 2019], formalize meta-learning as optimizing a shared parameter initialization that adapts via inner-loop gradient descent on support examples. Although expressive, these methods suffer from computational issues: calculating second-order metagradients is memory-prohibitive, and test-time adaptation mandates analytical backpropagation through unrolled graphs.

Memory-augmented architectures, such as MANN [Santoro et al., 2016] and SNAIL [Mishra et al., 2018], replace gradient descent with recurrent dynamics, accumulating task representations directly in episodic memory. Specifically, MANN utilizes an external memory matrix with content-based addressing to dynamically store and recall class bindings, while SNAIL combines temporal causal convolutions with attention mechanisms to pinpoint past exemplars across an episode. However, both approaches flatten inputs into 1D feature vectors—discarding native 2D spatial geometry—and rely on centralized memory controllers or attention operations that scale poorly with episode length and introduce artificial order dependence into unordered support sets. In modern regimes, large language and vision models exhibit emergent in-context learning [Brown et al., 2020, Garg et al., 2022, von Oswald et al., 2023]. However, in-context inference typically requires massive parameter footprints $( 1 0 ^ { 8 } – 1 0 ^ { 1 1 } )$ ), exhibits quadratic attention complexity, and lacks spatial locality priors.

Methods such as Matching Networks [Vinyals et al., 2016], Prototypical Networks [Snell et al., 2017], and Relation Networks [Sung et al., 2018] map inputs into metric feature spaces where classification reduces to nearest-neighbor or prototype distance comparisons.

## 2.2 Neural Cellular Automata: From Morphogenesis to General Computation

Neural Cellular Automata (NCAs) integrate classical cellular automata [von Neumann and Burks, 1966, Wolfram, 2002] with end-to-end differentiable neural components. Initially introduced to model biological morphogenesis, self-repair, and homeostatic pattern maintenance [Mordvintsev et al., 2020, Niklasson et al., 2021], NCAs demonstrate how uniform, strictly localized update rules can coordinate globally stable anatomical structures.

Beyond generative morphology, recent work has explored discriminative and computational NCA for mulations. Randazzo et al. [2020] demonstrated that NCAs can achieve global agreement for digit classification (Self-classifying MNIST), proving that local message passing can perform perceptual classification without centralized pooling. Concurrently, efforts toward general and universal cellular computation [Variengien et al., 2021, Bena et al.´ , 2025] have shown that continuous NCA state spaces can emulate discrete logic, matrix arithmetic, and routing operations.

To bridge low-level morphogenesis with high-level cognitive behavior, recent architectures have introduced modular memory and developmental hierarchies [Guichard et al., 2025b, Bena et al.´ , 2025, Pande and Grattarola, 2023]. These architectures partition the hidden channel into mutable and immutable states, with the development of said states being decoupled from each other. Building upon this, NCAs have applied developmental cellular dynamics to the Abstraction and Reasoning Corpus (ARC-AGI) [Guichard et al., 2025a, Xu and Miikkulainen, 2025], demonstrating that localized state-refinement dynamics are well-suited for abstract grid transformations. METALEARNNCA advances this trajectory by formulating a dual-NCA interaction loop where an Active-NCA is modulated by a learned Meta-NCA that translates task-level residual signals into spatial program fields.

## 2.3 Decentralized and Biologically Plausible Credit Assignment

Standard deep learning relies on global backpropagation, which exhibits well-documented biological implausibilities, including the weight-transport problem, non-local gradient flow, and global backward phases [Crick, 1989, Lillicrap et al., 2020]. Biologically plausible alternatives—such as Feedback Alignment [Lillicrap et al., 2016], Direct Feedback Alignment (DFA) [Nøkland, 2016], Predictive Coding [Whittington and Bogacz, 2019], and Equilibrium Propagation [Scellier and Bengio, 2017]—substitute exact backward transpose operations with local error projections or energy-relaxation phases.

## 3 Model Architecture

The MetaLearnNCA architecture performs few-shot spatial reasoning tasks through the dynamical coupling of two NCAs: an Active-NCA (f<sub>θ</sub>), which executes task inference conditioned on a spatial program, and a Meta-NCA $( \mathbf { g } _ { \phi } )$ , which iteratively infers and updates this spatial program by reading residual error signals. Both networks build upon the standard Neural Cellular Automata (NCA) formulation [Mordvintsev et al., 2020] and conditioned routing mechanisms [Guichard et al., 2025b, Bena et al.´ , 2025].

## 3.1 Base NCA Formulation

Both models share a common local transition backbone (BaseNCA). Let $\mathbf { s } ^ { ( t ) } \in \mathbb { R } ^ { B \times C _ { \mathrm { i n } } \times H \times W }$ denote the cellular grid state at discrete time-step t. The local state transition comprises a fixed spatial perception operation followed by a point-wise Multi-Layer Perceptron (MLP):

Perception Phase: Each channel of the state grid is padded with circular boundary conditions and convolved with four fixed $3 \times 3$ differential filters $\mathbf { K } = \{ K _ { \mathrm { i d e n t } } , K _ { \mathrm { s o b e l } , x } , K _ { \mathrm { s o b e l } , y } , K _ { \mathrm { l a p } } \}$ , computing identity, di rectional Sobel gradients, and discrete Laplacian operators (exact filter matrices are given in Appendix A.1). Applying K across all $C _ { \mathrm { i n } }$ channels yields a perceived feature representation $\mathbf { P } ( \mathbf { s } ^ { ( t ) } ) \in \mathbb { R } ^ { B \times 4 \bar { C _ { \mathrm { i n } } } \times H \times W }$

Update Phase: The perceived vector per cell is mapped to a state differential ds through a two-layer $1 \times 1$ convolutional network:

$$
d s = \mathbf { W } _ { 2 } \cdot \operatorname { R e L U } \left( \mathbf { W } _ { 1 } \mathbf { P } ( \mathbf { s } ^ { ( t ) } ) + \mathbf { b } _ { 1 } \right) ,\tag{1}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d _ { \mathrm { h i d d e n } } \times 4 C _ { \mathrm { i n } } } , \mathbf { W } _ { 2 } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times d _ { \mathrm { h i d d e n } } } \left( \mathrm { w i t h } \ d _ { \mathrm { h i d d e n } } = 2 5 6 \right)$ , and $\mathbf { W } _ { 2 }$ is initialized to zero without bias $( \mathbf { W } _ { 2 } = \mathbf { 0 } )$ to ensure stable identity transitions at initialization.

To simulate asynchronous biological cellular behavior, cell updates are modulated by a stochastic dropout mask:

$$
\begin{array} { r } { \begin{array} { r } { \mathbf { s } _ { \mathrm { d y n } } ^ { ( t + 1 ) } = \mathbf { s } _ { \mathrm { d y n } } ^ { ( t ) } + d s \odot \mathbf { M } _ { \mathrm { s t o c h } } , \quad \mathrm { w h e r e } \quad \mathbf { M } _ { \mathrm { s t o c h } } \sim \mathbf { B e r n o u l l i } ( 0 . 5 ) \in \{ 0 , 1 \} ^ { B \times 1 \times H \times W } . } \end{array} } \end{array}\tag{2}
$$

## 3.2 Active NCA

The Active-NCA $\mathbf { f } _ { \theta }$ maps visual inputs to task solutions conditioned on a persistent spatial program $\mathbf { S } .$ . Its total state is partitioned into three functional sets $( C _ { \mathrm { i n } } = 1 + C _ { s } + C _ { d } = 5 7 $ channels): (i) Observation

Channel $( C _ { \mathrm { i m g } } = 1 )$ , the normalized input image $\mathbf { X } \in [ - 1 , 1 ] ^ { B \times 1 \times H \times W }$ ; (ii) Static Program Channels $( C _ { s } = 2 4 )$ , the read-only spatial memory $\mathbf { S } \in \bar { \mathbb { R } ^ { B \times 2 4 \times H \times W } }$ written by the Meta-NCA; and (iii) Dynamic Reasoning Channels $( C _ { d } = 3 2 )$ , the read-write state $\mathbf { D } _ { \mathrm { a c t i v e } }$ , where channels $0 \ldots 9$ represent class logits maps Y<sup>ˆ</sup> and channels $1 0 \ldots 3 1$ serve as internal recurrent reasoning memory.

Concatenating the static program grid S with the dynamic state prior to perception allows the program channels to serve as spatially distributed pre-activation biases on upstream convolutional neurons. Because each cell computes local state differentials based on its immediate neighborhood, this continuous spatial bias directly routes the flow of information across the grid during inference without requiring centralized routing circuitry (see Appendix A.2 and Figure 5 for a detailed schematic of this conditioning mechanism).

## 3.3 Meta-NCA

The Meta-NCA $\mathbf { g } _ { \phi }$ acts as a decentralized optimizer. Rather than utilizing explicit gradient backpropagation during few-shot adaptation, it receives a spatial residual tensor $\mathbf { E } _ { \mathrm { p i x e l } } \in \bar { \mathbb { R } } ^ { B \times \bar { C } _ { \mathrm { l o s s } } \times \bar { H } \times W }$ defined by:

$$
\mathbf { E } _ { \mathrm { p i x e l } } = ( \mathbf { Y } _ { \mathrm { s p a t i a l } } - \mathrm { S o f t m a x } ( \hat { \mathbf { Y } } ) ) \odot \mathbf { M } _ { \mathrm { a l i v e } } ,\tag{3}
$$

where $\mathbf { Y } _ { \mathrm { s p a t i a l } } \ \in \ \{ 0 , 1 \} ^ { B \times 1 0 \times H \times W }$ is the one-hot target tensor broadcast spatially across all pixels, and $\mathbf { M } _ { \mathrm { a l i v e } } = \mathbb { I } ( \mathbf { X } > - 0 . 8 )$ is a binary foreground mask that isolates character strokes and prevents background pixels from generating penalizing error signals. The masking is also used during training when computing class logits, as in Randazzo et al. [2020].

The Meta-NCA state combines $\mathbf { E } _ { \mathrm { p i x e l } }$ with its internal dynamic memory $\mathbf { D } _ { \mathrm { m e t a } } \in \mathbb { R } ^ { B \times 3 2 \times H \times W } ( C _ { \mathrm { i n } } =$ $1 0 + 3 2 = 4 2$ channels). Over N internal recurrence steps, the Meta-NCA diffuses local error feedback across neighboring cells to update $\bf { D } _ { \mathrm { { m e t a } } }$ . The updated task program is extracted directly from the first $C _ { s } = 2 4$ dynamic channels.

## 3.4 Ensemble Interaction Loop

The full MetaLearnNCA framework coordinates the two models within a nested interaction loop (Figure 1; formal algorithmic outline provided in Appendix A.3, Algorithm 1):

1. Initialization: A shared spatial program grid is initialized randomly: $\mathbf { S } _ { \mathrm { t a s k } } \sim \mathcal { U } ( 0 , 1 ) ^ { 1 \times 2 4 \times H \times W }$ , and the meta dynamic state is cleared $\mathbf { ( D _ { \mathrm { { m e t a } } }  0 ) }$

2. Support Adaptation (K iterations): For each interaction step $k \in \{ 1 , \ldots , K \}$

(a) Broadcast $\mathbf { S _ { t a s k } }$ across the support batch: $\mathbf { S } \gets \mathrm { e x p a n d } ( \mathbf { S } _ { \mathrm { t a s k } } , B _ { \mathrm { s u p p } } )$

(b) Execute Active-NCA for 2N steps: $\mathbf { D } _ { \mathrm { s u p p } }  \mathbf { f } _ { \theta } ( \mathbf { X } _ { \mathrm { s u p p } } , \mathbf { S } , \mathrm { s t e p s } = 2 N )$

(c) Compute the residual $\mathbf { E } _ { \mathrm { p i x e l } }$ over the support set.

(d) Execute Meta-NCA for N steps: $\mathbf { D } _ { \mathrm { m e t a } }  \mathbf { g } _ { \phi } ( \mathbf { E } _ { \mathrm { p i x e l } } , \mathbf { D } _ { \mathrm { m e t a } } , \mathrm { s t e p s } = N )$

(e) Aggregate support programs via batch mean-pooling: $\begin{array} { r } { \mathbf { S } _ { \mathrm { t a s k } }  \frac { 1 } { B _ { \mathrm { s u p p } } } \sum _ { i = 1 } ^ { B _ { \mathrm { s u p p } } } \mathbf { D } _ { \mathrm { m e t a } , i } ^ { [ 0 : C _ { s } , : , : ] } } \end{array}$

3. Query Inference: The adapted program $\mathbf { S } _ { \mathrm { t a s k } }$ is frozen and broadcast to the query set: $\mathbf { S } _ { \mathrm { q r y } } \gets$ $\mathrm { e x p a n d } ( { \bf S } _ { \mathrm { t a s k } } , B _ { \mathrm { q r y } } )$ . The Active-NCA then executes inference on $\mathbf { X } _ { \mathrm { q r y } }$ for 2N steps to produce the final predictions $\mathbf { D } _ { \mathrm { q r y } }$

![](images/f5eff02fffee6cd251d6d0c76d808c9c28994e952022847135d10a1139c46568.jpg)  
Figure 1: The MetaLearnNCA interaction loop: Active-NCA computes forward task predictions on the support set, while Meta-NCA translates the residual error maps into an updated spatial program $\mathbf { S _ { t a s k } }$

Hypothetical Mechanics of Distributed Memory Superposition. Unlike metric meta-learners that store isolated 1D prototype vectors, METALEARNNCA encodes the full multi-class task representation within a single, superposed spatial memory manifold $S _ { \mathrm { t a s k } } \in \mathbb { R } ^ { C _ { s } \times H \times W }$ . In high-dimensional cellular lattices $( C _ { s } \times H \times W = 1 8 , 8 1 6 )$ , class-specific steering fields written by the Meta-NCA could occupy quasiorthogonal subspaces. Batch mean-pooling in step (e) could acts as an additive holographic superposition. This was not empirically tested but could explain why batch mean pooling does not destructively interfere with class specific artifacts.

## 4 Training Methodology

The parameters $\Theta = \{ \theta , \phi \}$ of both the Active-NCA and Meta-NCA are optimized concurrently through the unrolled adaptation trajectory (Algorithm 1, Appendix A.3), heavily inspired by FOMAML [Finn et al., 2017].

Meta-Objective Formulation: The query loss $\mathcal { L } _ { \mathrm { q r y } }$ is computed as a stroke-masked spatial Cross-Entropy loss. For each query instance i, raw logit maps $\hat { \mathbf { Y } } _ { \mathrm { q r y } , i } \in \mathbb { R } ^ { C _ { \mathrm { l o s s } } \times H \times W }$ are supervised by the scalar groundtruth label $y _ { \mathrm { q r y } , i }$ broadcast spatially across all grid coordinates $( u , v )$ belonging to the active stroke mask ${ \bf { M } } _ { \mathrm { { q r y } } , i } .$

$$
\mathcal { L } _ { \mathrm { q r y } } = \frac { \sum _ { i = 1 } ^ { B _ { \mathrm { q r y } } } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } - \log \left( \frac { \exp ( \hat { \mathbf { Y } } _ { \mathrm { q r y } , i } ( y _ { \mathrm { q r y } , i } , u , v ) ) } { \sum _ { c = 0 } ^ { C _ { \mathrm { l o s s } } - 1 } \exp ( \hat { \mathbf { Y } } _ { \mathrm { q r y } , i } ( c , u , v ) ) } \right) \cdot \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) } { \sum _ { i = 1 } ^ { B _ { \mathrm { q r y } } } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) + \epsilon }\tag{4}
$$

During discrete test-time evaluation, predictions $\hat { y } _ { i }$ are extracted by globally average pooling logits across all active stroke pixels:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c \in \{ 0 , \ldots , C _ { \mathrm { l o s s } } - 1 \} } \frac { \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \hat { \mathbf { Y } } _ { \mathrm { q r y } , i } ( c , u , v ) \cdot \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) } { \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) + \epsilon } .\tag{5}
$$

Temporal Jittering for Robustness: To prevent the cellular dynamics from overfitting to exact unroll lengths, we apply random step-length jittering during meta-training $( N ^ { \prime } \in [ 1 6 , 1 9 ]$ and $K ^ { \prime } \in [ 3 , 5 ] )$ . The total effective minimal interaction steps $( K \times 3 N = 3 \times 4 8 = 1 4 4 )$ substantially exceed the canvas diameter $( H = W = 2 8 $ , grid diagonal ≈ 39.6), ensuring sufficient time for information to propagate across NCAs (with kernels of size 3, the speed of light within an NCA is 1 pixel per step).

## 5 Experiments

We evaluate MetaLearnNCA on few-shot character recognition using the Omniglot dataset (964 background classes) for training. All images are resized to 28 × 28, inverted such that background pixels equal 0.0 and character strokes equal 1.0, and normalized to [−1, 1].

Episode Sampling: In each meta-training iteration, we randomly sample a 10-way, 1-support, 5-query task $( B _ { \mathrm { s u p p } } = 1 0 , B _ { \mathrm { q r y } } = 5 0 )$ . True Omniglot class IDs are mapped dynamically to surrogate class indices $\{ 0 , \ldots , 9 \}$

Optimization Details: The joint model is trained for E = 5000 meta-epochs with a meta-batch size of $B = 3 2$ tasks using AdamW $( l r = 1 0 ^ { - 3 }$ , weight decay = 10<sup>−5</sup>). Gradients are clipped to a maximum $\ell _ { 2 }$ norm of 1.0.

## 6 Results

We evaluate METALEARNNCA along three dimensions: (i) few-shot out-of-distribution (OOD) crossdataset transfer, (ii) spatial fault tolerance and damage resilience, and (iii) qualitative visual inspection of the learned spatial program fields. We present further results and analysis in Appendices D, E, F, G, H I.

## 6.1 Out Of Distribution Offline Learning

To evaluate whether the learned cellular optimizer (META-NCA) acquires a generalizable error-diffusion dynamic rather than memorizing in-distribution character templates, we evaluate models meta-trained solely on Omniglot across five completely unseen out-of-distribution visual domains: MNIST, USPS, KMNIST, Fashion-MNIST, and MedMNIST Organ (referred to as MedMNIST). Full testing algorithm can be found in Appendix B.

Table 1 evaluates METALEARNNCA against canonical metric- and optimization-based baselines across 1-shot, 5-shot, and 10-shot adaptation regimes over 10 independent test splits (1 random checkpoint, 10 testing seeds). While FOMAML and ProtoNet maintain a marginal lead in-domain on Omniglot (≈ 97%–98% at 10-shot). In contrast, METALEARNNCA demonstrates somewhat better cross-domain transfer: it outper forms all baselines across all shot counts on MNIST (reaching 89.71% at 10-shot, +10.54% over ProtoNet), cursive KMNIST (51.26%, +15.03% over FOMAML), and Fashion-MNIST (54.40%, +3.87% over FO-MAML). Furthermore, in a 1-shot domain setting, METALEARNNCA achieves the highest accuracy across 4 of 5 unseen target domains (including 53.26% on USPS, beating FOMAML by +10.86%). FOMAML outperformed MetaLearnNCA across all shots on MedMNIST (+4.28%, +16.51%, +7.81%) and on 1- and 5- shot on USPS (+7.81%, +4.72%). Crucially, METALEARNNCA achieves this transfer purely through forward cellular dynamics.

Table 1: Few-shot adaptation performance (Accuracy % ± 95% CI) across 1-shot, 5-shot, and 10-shot regimes on in-domain Omniglot and cross-domain out-of-distribution transfer benchmarks. All models (except ours) employ the canonical Conv4 backbone on $2 8 \times 2 8$ inputs.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">Gradient-Free Baselines</td><td rowspan="2">Gradient-Based FOMAML</td><td rowspan="2">Ours MetaLearnNCA</td></tr><tr><td></td><td></td><td>Shot MatchingNet RelationNet</td><td>ProtoNet</td></tr><tr><td rowspan="3">Omniglot (In-Dist)</td><td>1</td><td> $6 3 . 4 7 \pm 7 . 7 9$ </td><td> $8 4 . 8 4 \pm 3 . 7 4$ </td><td> ${ \bf 9 2 . 2 6 \pm 3 . 1 5 }$ </td><td> $8 7 . 8 4 \pm 3 . 6 7$ </td><td> $8 7 . 9 5 \pm 4 . 1 6$ </td></tr><tr><td>5</td><td> $7 5 . 4 7 \pm 4 . 5 4$ </td><td> $9 2 . 7 3 \pm 3 . 0 9$ </td><td> $9 6 . 4 0 \pm 3 . 0 9$ </td><td> ${ \bf 9 6 . 9 3 \pm 1 . 4 1 }$ </td><td> $9 5 . 2 0 \pm 2 . 1 3$ </td></tr><tr><td>10</td><td> $7 7 . 4 0 \pm 6 . 3 7$ </td><td> $9 3 . 2 0 \pm 2 . 6 5$ </td><td> $9 7 . 4 0 \pm 1 . 2 7$ </td><td> ${ \bf 9 8 . 8 0 \pm 0 . 9 4 }$ </td><td> $9 6 . 1 2 \pm 0 . 7 0$ </td></tr><tr><td rowspan="3">MNIST</td><td>1</td><td> $4 4 . 7 9 \pm 4 . 9 3$ </td><td> $5 1 . 1 7 \pm 2 . 6 6$ </td><td> $5 5 . 3 2 \pm 1 . 8 7$ </td><td> $4 8 . 1 6 \pm 2 . 9 8$ </td><td> ${ \bf 6 5 . 0 2 \pm 3 . 0 7 }$ </td></tr><tr><td>5</td><td> $5 8 . 9 3 \pm 1 . 3 4$ </td><td> $6 3 . 7 4 \pm 1 . 3 9$ </td><td> $7 5 . 1 7 \pm 1 . 5 6$ </td><td> $7 0 . 8 0 \pm 1 . 7 2$ </td><td> ${ \bf 8 4 . 1 2 \pm 1 . 5 7 }$ </td></tr><tr><td>10</td><td> $5 8 . 6 6 \pm 1 . 1 5$ </td><td> $6 7 . 3 0 \pm 0 . 6 6$ </td><td> $7 9 . 1 7 \pm 1 . 4 8$ </td><td> $7 2 . 3 7 \pm 2 . 0 1$ </td><td> $\mathbf { 8 9 . 7 1 \pm 0 . 4 3 }$ </td></tr><tr><td rowspan="3">USPS</td><td>1</td><td> $3 0 . 2 3 \pm 3 . 3 3$ </td><td> $2 4 . 1 4 \pm 3 . 4 6$ </td><td> $3 5 . 5 0 \pm 3 . 7 4$ </td><td> $4 2 . 4 0 \pm 4 . 6 2$ </td><td> ${ \bf 5 3 . 2 6 \pm 5 . 9 9 }$ </td></tr><tr><td>5</td><td> $3 4 . 3 9 \pm 3 . 0 3$ </td><td> $3 5 . 4 1 \pm 2 . 6 9$ </td><td> $5 1 . 0 7 \pm 2 . 0 4$ </td><td> ${ \bf 6 4 . 9 1 \pm 2 . 1 3 }$ </td><td> $5 7 . 1 0 \pm 4 . 2 6$ </td></tr><tr><td>10</td><td> $3 6 . 5 8 \pm 3 . 0 2$ </td><td> $3 8 . 8 1 \pm 1 . 1 9$ </td><td> $5 5 . 5 4 \pm 1 . 8 9$ </td><td> ${ \bf 6 9 . 7 5 \pm 2 . 1 6 }$ </td><td> $6 5 . 0 2 \pm 3 . 1 0$ </td></tr><tr><td rowspan="3">Fashion-MNIST</td><td>1</td><td> $2 5 . 2 4 \pm 2 . 8 4$ </td><td> $2 0 . 8 2 \pm 2 . 5 9$ </td><td> $2 2 . 2 6 \pm 1 . 7 5$ </td><td> $3 1 . 3 9 \pm 3 . 2 1$ </td><td> ${ \bf 4 0 . 4 3 \pm 3 . 1 1 }$ </td></tr><tr><td>5</td><td> $3 1 . 6 8 \pm 0 . 8 2$ </td><td> $2 5 . 4 3 \pm 0 . 7 5$ </td><td> $2 9 . 6 6 \pm 0 . 9 1$ </td><td> $4 7 . 4 1 \pm 1 . 8 8$ </td><td> ${ \bf 5 1 . 3 4 \pm 2 . 6 7 }$ </td></tr><tr><td>10</td><td> $3 2 . 9 9 \pm 0 . 7 5$ </td><td> $2 5 . 9 1 \pm 1 . 5 8$ </td><td> $3 1 . 6 3 \pm 1 . 2 9$ </td><td> $5 0 . 5 3 \pm 1 . 8 2$ </td><td> ${ \bf 5 4 . 4 0 \pm 1 . 2 2 }$ </td></tr><tr><td rowspan="3">KMNIST</td><td>1</td><td> $1 7 . 1 5 \pm 1 . 4 6$ </td><td> $2 0 . 0 4 \pm 1 . 9 6$ </td><td> $2 2 . 5 7 \pm 2 . 3 5$ </td><td> $2 2 . 7 6 \pm 2 . 0 1$ </td><td> ${ \bf 3 0 . 2 8 \pm 3 . 1 4 }$ </td></tr><tr><td>5</td><td> $2 3 . 1 5 \pm 1 . 0 3$ </td><td> $2 6 . 0 6 \pm 1 . 6 0$ </td><td> $3 0 . 0 9 \pm 2 . 0 5$ </td><td> $3 3 . 6 6 \pm 1 . 6 6$ </td><td> $\mathbf { 4 1 . 6 5 \pm 1 . 8 6 }$ </td></tr><tr><td>10</td><td> $2 5 . 6 1 \pm 1 . 3 8$ </td><td> $2 9 . 4 0 \pm 1 . 1 3$ </td><td> $3 5 . 4 2 \pm 1 . 4 2$ </td><td> $3 6 . 2 3 \pm 1 . 1 8$ </td><td> ${ \bf 5 1 . 2 6 \pm 1 . 2 7 }$ </td></tr><tr><td rowspan="3">MedMNIST</td><td>1</td><td> $1 7 . 4 7 \pm 3 . 0 0$ </td><td> $1 0 . 7 9 \pm 3 . 8 3$ </td><td> $2 0 . 8 6 \pm 2 . 6 6$ </td><td> $\mathbf { 2 2 . 8 2 \pm 4 . 7 4 }$ </td><td> $1 8 . 5 4 \pm 2 . 1 6$ </td></tr><tr><td>5</td><td> $2 6 . 6 1 \pm 3 . 1 9$ </td><td> $1 2 . 2 4 \pm 3 . 5 1$ </td><td> $2 4 . 8 5 \pm 2 . 3 4$ </td><td> $\mathbf { 3 5 . 9 1 \pm 2 . 3 0 }$ </td><td> $1 9 . 7 6 \pm 3 . 8 6$ </td></tr><tr><td>10</td><td> $2 7 . 8 8 \pm 2 . 8 0 $ </td><td> $1 1 . 8 6 \pm 4 . 7 0$ </td><td> $2 5 . 9 9 \pm 2 . 1 4$ </td><td> $\mathbf { 3 4 . 2 4 \pm 1 . 3 1 }$ </td><td> $2 6 . 4 3 \pm 2 . 2 2$ </td></tr><tr><td colspan="2">Test-Time Gradients? Parameters</td><td>No  $\approx 1 1 5 \mathrm { k }$ </td><td>No  $\approx 2 2 5 \mathrm { k }$ </td><td>No  $\approx 1 1 5 \mathrm { k }$ </td><td> $Y e s \left( 5 \ : s t e p s \right)$   $\approx 1 1 5 \mathrm { k }$ </td><td>No  $\approx \mathbf { 1 1 8 k }$ </td></tr></table>

To measure end-to-end (training and testing) variability, Figure 2 plots test-time adaptation trajectories across offline cellular optimization repeats $( k \in \{ 0 , \ldots , 5 \} )$ in the 10-shot regime, aggregated across 42 independent meta-training runs and 42 independent testing seeds (One random test seed per random training seed). Crucially, the 95% confidence interval bands demonstrate that METALEARNNCA converges consistently regardless of the training or testing seed. Across all domains, the coupled cellular dynamics display an emergent two-phase behavior: remaining flat at chance level (≈ 10%) at k = 0 and k = 1, followed by a sharp transition at $k = 2 ,$ and plateauing stably between k = 3 and k = 4 without diverging under extended recurrence (except for USPS).

![](images/711448cfb3d2ae730751f407945fad228b148b2caff2c217b23ae57c3f6be666.jpg)  
Figure 2: Few-shot offline 10-shot adaptation trajectories of METALEARNNCA evaluated on out-ofdistribution character datasets across offline cellular optimization repeats $( k \in \{ 0 , \ldots , 5 \} )$ . Solid curves depict mean accuracy across 42 independent meta-training runs and 42 independent testing seeds, with shaded regions denoting 95% confidence intervals.

## 6.2 Program Damage

A defining characteristic of biological cellular ensembles is structural fault tolerance. We evaluate whether the continuous spatial representation $S _ { \mathrm { t a s k } }$ retains this physical property under severe localized damage.

Following support adaptation, we corrupt the spatial program grid $S _ { \mathrm { t a s k } }$ by injecting circular patches of random noise of radius $r = 4$ pixels $( A _ { \mathrm { d a m a g e } } \approx 5 0 . 3 \ : \mathrm { p x ^ { 2 } }$ , spanning $\approx 6 . 4 \%$ of the total $2 8 \times 2 8$ grid area per circle across all 24 channels). We evaluate test classification accuracy as a function of the number of damaged patches $( d \in \{ 0 , \ldots , 6 \} )$ ).

As depicted in Figure 3, with a single localized circular puncture $( d = 1 )$ , METALEARNNCA retains 93.4% classification accuracy (a marginal $\approx 4 . 6 \%$ drop relative to the uncorrupted 98.0% baseline). Even when subjected to two non-overlapping punctures $( d \ = \ 2 ,$ destroying over 12% of the spatial memory footprint), performance remains resilient at an average 81.8%. Variance between runs, however, increases greatly. Performance degrades smoothly as damage scales $( d \geq 3 )$ , dropping toward chance only when larger fractions of the grid are physically destroyed.

## 6.3 Hidden Channel and Program Field Visualizations

We visualize a small section of the hidden channels and program on the MNIST dataset. We chose to visualize MNIST as the characters are easier to recognize.

Figure 4 illustrates representative hidden reasoning channels from the Active-NCA and program channels from the Meta-NCA during inference on Mnist. The dynamic reasoning channels exhibit chaotic behavior, possibly representing a higher-order feature of the classification task. The program channels demonstrate functional specializations such as: Spot attention (channel 5), positive general shape matching (channel 13), negative shape matching (channel 21), and more complex compound features (channel 23). See Appendix C for the complete 46-channel atlas.

![](images/a3ce06fc89ea259fe08dcfe982764dd2a3fc4ffc9a67dee777ab90e644b02c69.jpg)

Figure 3: Resilience of MetaLearnNCA to physical spatial damage: test accuracy as a function of circular noise patches $( r = 4 { \mathrm { p x } } )$ injected into the static program grid $S _ { \mathrm { t a s k } }$  
![](images/522e7f767a9298574b32d5fce46ec677ecb3432ebd5869c8c5f541f4757d2d4f.jpg)  
Figure 4: Representative spatial and program channels during MNIST inference. (see Appendix C for the complete 46-channel atlas).

## 7 Discussion

Beyond eliminating test time gradients, METALEARNNCA shifts the representational paradigm of few-shot learning by encoding task knowledge as a continuous, spatially structured dynamical field rather than parametric weight perturbations or global 1D vector embeddings. Crucially, our framework embodies a decentralized realization of localized credit assignment that bypasses the non-local weight-transport requirements of backpropagation. Instead of back-propagating errors through analytical layer inversion, task adaptation modifies the spatial states used by the networks during forward computation, while the weights remain fixed. This demonstrates how effective task adaptation can emerge on non-Von Neumann substrates with explicit aggregation operations.

From a neurobiological perspective, METALEARNNCA can be viewed as a speculative spatially grounded abstraction of biological cortical dynamics. Each cell on the lattice functions analogously to an individual neuron or cortical microcolumn, with a local $3 \times 3$ perception that models the recurrent connectivity among local neurons. Within this microcircuit, dynamic reasoning channels emulate fast-timescale electrophysio logical variables such as membrane potentials and intracellular signaling cascades, while the persistent spatial program channels (S) operate as slow neuromodulatory fields or synaptic trace maps that alter upstream neural excitability without modifying structural weights. In this view, the Meta-NCA acts as a feedback or subcortical neuromodulatory network that detects sensory prediction errors and broadcasts chemical or electrical signals throughout the tissue, thereby regulating the attractor dynamics of the primary network entirely through recurrent activations.

Several limitations of this study highlight important avenues for future work. First, our evaluation relies on character and silhouette benchmarks where visual concepts exhibit a strong spatial center bias, which naturally aids coordinate-dependent steering; extending this formulation to unconstrained scenes with translation, scale variance, and background clutter remains to be demonstrated. Second, our core architectural hyperparameters—including channel partitions $( C _ { s } = 2 4 , C _ { d } = 3 2 )$ , MLP hidden widths $( d _ { \mathrm { h i d d e n } } = 2 5 6 )$ inner steps $( N = 1 6 )$ , and outer test repeats $( K = 5 )$ —were selected heuristically rather than through systematic optimization sweeps. Third, while test-time adaptation is gradient-free and activation memory is independent of unroll length during inference, unrolling recurrent trajectories makes meta-training computationally demanding in wall-clock time and backpropagation memory. The mechanism for modulation (appending the program channels to the NCA) remains limited in functional expressivity. Scaling directly to higher-resolution RGB canvases is constrained by the cellular speed of light. Finally, the experiments do not isolate the benefits of the cellular updater, spatial memory, and foreground masking, or establish a runtime advantage. Future work will investigate Graph Neural Cellular Automata (GNCAs) over flexible topologica graphs, developing methods for more expressive programs, and apply interacting NCAs to abstract program induction on the Abstraction and Reasoning Corpus (ARC-AGI).

## 8 Conclusion

We introduced MetaLearnNCA, an offline meta-learning framework that achieves few-shot adaptation through the dynamical interaction of coupled Neural Cellular Automata without test-time backpropagation. By formulating task representations as continuous 2D spatial memory fields and computing credit assignment via localized updates to a residual error map, METALEARNNCA bridges the gap between gradientfree meta-learning and decentralized cellular self-organization. Our framework matches established baselines in-distribution while achieving gains on some OOD transfer tasks. METALEARNNCA demonstrates that gradient-free learning-to-learn can emerge from decentralized cellular dynamics on non-von Neumann computing substrates.

## AI use statement

In this work, generative AI assistance was used for LaTeX syntax restructuring, template formatting, and phrasing refinement. All theoretical derivations, algorithmic designs, experimental implementations, and conclusions were conceived, verified, and validated by the author(s).

## Reproducibility Statement

Complete PyTorch code, training scripts, and evaluation routines for both METALEARNNCA and all baselines are open-sourced via an anonymous repository at: https://github.com/etimush/MetaLearnNCA.

## Acknowledgment

This work was supported by MishMash - Research Council of Norway, GrantId: 357438.

## References

Gabriel Bena, Maxence Faldor, Dan F. M. Goodman, and Antoine Cully. A path to universal neural cellular´ automata, 2025. URL https://arxiv.org/abs/2505.13058.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 1877–1901, 2020.

Francis Crick. The recent excitement about neural networks. Nature, 337(6203):129–132, 1989.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning (ICML), pages 1126–1135, 2017.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 30583–30598, 2022.

Etienne Guichard and Stefano Nichele. Neural cellular automata learn general features in their hidden channels, 2026. URL https://arxiv.org/abs/2609.21870.

Etienne Guichard et al. Arc-nca: Towards developmental solutions to the abstraction and reasoning corpus. arXiv preprint, 2025a.

Etienne Guichard et al. Engramnca: a neural cellular automaton model of memory transfer. arXiv preprint, 2025b.

Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5149–5169, 2021.

Michael Levin. The computational boundary of a “self”: developmental bioelectricity drives multicellularity and scale-free cognition. Frontiers in Psychology, 10:2688, 2019.

Zhenguo Li, Fengwei Zhou, Fei Chen, and Hang Li. Meta-sgd: Learning to learn quickly for few-shot learning. 2017.

Timothy P Lillicrap, Daniel Cownden, Douglas B Tweed, and Colin J Akerman. Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7(1):11140, 2016.

Timothy P Lillicrap, Adam Santoro, Luke Marris, Colin J Akerman, and Geoffrey Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335–346, 2020.

Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive meta-learner. In International Conference on Learning Representations (ICLR), 2018.

Milton L. Montero, Elias Najarro, Jakob Schauser, and Sebastian Risi. Learning developmental scaffoldings to guide self-organisation, 2026. URL https://arxiv.org/abs/2605.14998.

Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. Growing neural cellular automata. Distill, 5(2):e23, 2020.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.

Eyvind Niklasson, Alexander Mordvintsev, Ettore Randazzo, and Michael Levin. Self-organising textures. Distill, 6(2):e00027, 2021.

Arild Nøkland. Direct feedback alignment provides learning in deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 29, 2016.

Ritu Pande and Daniele Grattarola. Hierarchical neural cellular automata. In ALIFE 2023: Ghost in the Machine: Proceedings of the 2023 Artificial Life Conference, page 20, 07 2023. doi: 10.1162/isal a 00601. URL https://doi.org/10.1162/isal\_a\_00601.

Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

Ettore Randazzo, Alexander Mordvintsev, Eyvind Niklasson, Michael Levin, and Sam Greydanus. Selfclassifying mnist digits: Achieving distributed coordination with neural cellular automata. Distill, 5(8): e27, 2020.

Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning (ICML), pages 1842–1850, 2016.

Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Comparing gradient-descent and biologically plausible neural networks. Frontiers in Computational Neuroscience, 11:24, 2017.

Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1199–1208, 2018.

Alexandre Variengien, Stefano Nichele, Tom Glover, and Sidney Pontes-Filho. Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent, 2021. URL https:// arxiv.org/abs/2106.15240.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 29, 2016.

John von Neumann and Arthur W Burks. Theory of self-reproducing automata. University of Illinois Press Urbana, 1966.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, Joao Sacramento, Alexander Mordvintsev, An-˜ drey Zhmoginov, and Julian Zilly. Transformers learn in-context by gradient descent. In International Conference on Machine Learning (ICML), pages 35151–35174, 2023.

James CR Whittington and Rafal Bogacz. Theories of error back-propagation in the brain. Trends in Cognitive Sciences, 23(3):235–250, 2019.

Stephen Wolfram. A new kind of science, volume 5. Wolfram media Champaign, IL, 2002.

Kevin Xu and Risto Miikkulainen. Neural cellular automata for arc-agi, 2025. URL https://arxiv. org/abs/2506.15746.

## A Extended Architectural and Algorithmic Details

## A.1 Explicit Perception Filter Matrices

The four fixed $3 \times 3$ differential filters $\mathbf { K } = \{ K _ { \mathrm { i d e n t } } , K _ { \mathrm { s o b e l } , x } , K _ { \mathrm { s o b e l } , y } , K _ { \mathrm { l a p } } \}$ used in the perception phase (Section 3.1) are defined as:

$$
K _ { \mathrm { i d e n t } } = \left[ { \begin{array} { l l l } { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } \end{array} } \right] , \quad K _ { \mathrm { s o b e l , x } } = \left[ { \begin{array} { l l l } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} } \right] , \quad K _ { \mathrm { s o b e l , y } } = K _ { \mathrm { s o b e l , x } } ^ { \top } , \quad K _ { \mathrm { l a p } } = \left[ { \begin{array} { l l l } { 1 } & { 1 } & { 1 } \\ { 1 } & { - 8 } & { 1 } \\ { 1 } & { 1 } & { 1 } \end{array} } \right] .\tag{6}
$$

## A.2 Single-Cell Conditioning Schematic

![](images/60b5c7769599846c6f5adf7ee0f87b07087f26fa7792b29a1e1a91c88096a455.jpg)  
Figure 5: Theoretical overview of the conditioning mechanism: the static spatial program $\mathbf { S }$ acts as a localized pre-activation bias, dynamically modulating information flow through the Active-NCA transition operator.

## A.3 Meta-NCA Meta-Training Algorithm

```latex
Algorithm 1 Meta-NCA Meta-Training Loop
Require: Dataset D, Meta-batch size B, Epochs E, Base Inner iterations $K = 3 ,$ , Base inner steps $N = 1 6 .$
Learning rate $\eta = 1 0 ^ { - 3 }$
1: Initialize Active-NCA $\mathbf { f } _ { \theta }$ and $\mathbf { M e t a - N C A } \mathbf { g } _ { \phi }$
2: Initialize AdamW optimizer for joint parameters $\Theta = \{ \theta , \phi \}$
3: for epoch = 1 to E do
4: $\nabla _ { \Theta } \mathcal { L } \gets 0$
5: for task $b = 1$ to B do
6: Sample episode $( \mathbf { X } _ { \mathrm { s u p p } } , \mathbf { y } _ { \mathrm { s u p p } } ) , ( \mathbf { X } _ { \mathrm { q r y } } , \mathbf { y } _ { \mathrm { q r y } } ) \sim \mathcal { D }$
7: Initialize program $\mathbf { S } _ { \mathrm { t a s k } } \sim \bar { \mathcal { U } } ( 0 , 1 ) ^ { 1 \times C _ { s } \times \bar { H } \times W }$ and meta-state $\mathbf { D } _ { \mathrm { { m e t a } } }  \mathbf { 0 }$
8: Sample randomized step horizons: $N ^ { \prime } \sim \mathcal { U } \{ N , N + 3 \} , K ^ { \prime } \sim \mathcal { U } \{ K , K + 2 \}$
// Inner Adaptation Loop (Support Set)
9: for $k = 1$ to $K ^ { \prime }$ do
10: $\mathbf { S } \gets \mathrm { e x p a n d } ( \mathbf { S } _ { \mathrm { t a s k } } , B _ { \mathrm { s u p p } } )$
11: $ { \mathbf { D } } _ { \mathrm { s u p p } }   { \mathbf { f } } _ { \boldsymbol { \theta } } (  { \mathbf { X } } _ { \mathrm { s u p p } } ,  { \mathbf { S } } ,  { \mathrm { s t e p s } } = 2 N ^ { \prime } )$
12: $\mathbf { E } _ { \mathrm { p i x e l } }  ( \mathbf { Y } _ { \mathrm { s u p p } } - \mathrm { S o f t m a x } ( \mathbf { D } _ { \mathrm { s u p p } } ^ { [ 0 : C _ { \mathrm { l o s s } } ] } ) ) \odot \mathbb { I } ( \mathbf { X } _ { \mathrm { s u p p } } > - 0 . 8 )$
13: $\mathbf { D } _ { \mathrm { m e t a } }  \dot { \mathbf { g } _ { \phi } } ( \mathbf { E } _ { \mathrm { p i x e l } } , \mathbf { D } _ { \mathrm { m e t a } } , \mathrm { s t e p s } = N ^ { \prime } )$
14: $\begin{array} { r } { \mathbf { S } _ { \mathrm { t a s k } }  \frac { 1 } { B _ { \mathrm { s u p p } } } \sum _ { i = 1 } ^ { B _ { \mathrm { s u p p } } } \mathbf { D } _ { \mathrm { m e t a } , i } ^ { [ 0 : C _ { s } ] } } \end{array}$
15: end for
// Query Evaluation & Meta-Loss
16: $\mathbf { S } _ { \mathrm { q r y } }  \mathrm { e x p a n d } ( \mathbf { S } _ { \mathrm { t a s k } } , B _ { \mathrm { q r y } } )$
17: $\mathbf { D } _ { \mathrm { q r y } }  \mathbf { f } _ { \boldsymbol { \theta } } ( \mathbf { X } _ { \mathrm { q r y } } , \mathbf { S } _ { \mathrm { q r y } } ,$ steps = 2N)
18: $\begin{array} { r } { \hat { \mathbf { Y } } _ { \mathrm { q r y } }  \mathbf { D } _ { \mathrm { q r y } } ^ { [ 0 : C _ { \mathrm { l o s s } } ] } , \quad \mathbf { M } _ { \mathrm { q r y } }  \mathbb { I } ( \mathbf { X } _ { \mathrm { q r y } } > - 0 . 8 ) } \end{array}$
19: Compute masked spatial Cross-Entropy:
$\begin{array} { r }  \int _ { \mathrm { . c . e . } } \Big \langle - \frac { \sum _ { i = 1 } ^ { B _ { \ P \mathrm { y } } } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \ell _ { \mathrm { C E } } \left( \hat { \mathbf { Y } } _ { \ P \mathrm { f y } , i } ( \cdot , u , v ) , y _ { \mathrm { q r y } , i } \right) \cdot \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) } \end{array}$
qry $\begin{array} { r } { \sum _ { i = 1 } ^ { B _ { \mathrm { q r y } } } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } \mathbf { M } _ { \mathrm { q r y } , i } ( u , v ) + \epsilon } \end{array}$
20: Accumulate task meta-gradient: $\begin{array} { r } { \nabla _ { \Theta } \mathcal { L }  \nabla _ { \Theta } \mathcal { L } + \frac { 1 } { B } \nabla _ { \Theta } \mathcal { L } _ { \mathrm { q r y } } } \end{array}$
21: end for
22: Clip gradients: $\| \nabla _ { \theta } \| _ { 2 } \leq 1 . 0 , \quad \| \nabla _ { \phi } \| _ { 2 } \leq 1 . 0$
23: Update parameters: $\Theta  \mathbf { A }$ damW $\mathopen { } \mathclose \bgroup ( \Theta , \nabla _ { \Theta } \mathcal { L } , \eta _ { \cdot }$ , weight decay $= 1 0 ^ { - 5 } )$
24: end for
25: return Optimized parameters $\theta ^ { * } , \phi ^ { * }$
```

## B Testing Algorithm

Algorithm 2 METALEARNNCA Few-Shot Offline Inference and Testing Loop   
Require: Optimized Active-NCA parameters $\theta ^ { * }$ , Meta-NCA parameters $\phi ^ { * }$   
Require: Target test dataset $\mathcal { D } _ { \mathrm { t e s t } } ,$ Number of classes $N _ { \mathrm { w a y } }$ , Support shot $K _ { \mathrm { s h o t } }$   
Require: Base recurrence steps $N = 1 9 .$ , Optimization repeats $K _ { \mathrm { r e p e a t s } } = 5$   
Require: Optional test-time program spatial displacement $\Delta _ { \mathrm { s h i f t } } \geq 0$ (default: 0)   
1: Sample test episode: $( X _ { \mathrm { s u p p } } , y _ { \mathrm { s u p p } } ) , ( X _ { \mathrm { q r y } } , y _ { \mathrm { q r y } } ) \sim \mathcal { D } _ { \mathrm { t e s t } }$ ▷ $B _ { \mathrm { s u p p } } = N _ { \mathrm { w a y } } \times K _ { \mathrm { s h o t } }$   
2: Compute foreground stroke masks:   
3: $M _ { \mathrm { s u p p } }  \mathbb { I } ( X _ { \mathrm { s u p p } } > - 0 . 8 ) , \quad M _ { \mathrm { q r y } }  \mathbb { I } ( X _ { \mathrm { q r y } } > - 0 . 8 )$   
4: Initialize random spatial program grid: $S _ { \mathrm { t a s k } } ^ { ( 0 ) } \sim \stackrel { \textstyle \sim } { \mathcal { U } } ( 0 , 1 ) ^ { 1 \times C _ { s } \times H \times W }$   
5: $S _ { \mathrm { e v a l } } ^ { ( 0 ) }  \mathrm { r o l l } ( S _ { \mathrm { t a s k } } ^ { ( 0 ) } , ( \Delta _ { \mathrm { s h i f t } } , \Delta _ { \mathrm { s h i f t } } ) )$   
6: $S _ { \mathrm { q r y } }  \mathrm { e x p a n d } ( S _ { \mathrm { e v a l } } ^ { ( 0 ) } , B _ { \mathrm { q r y } } )$   
7: $D _ { \mathrm { q r y } }  f _ { \theta ^ { * } } ( X _ { \mathrm { q r y } } , S _ { \mathrm { q r y } } , \mathrm { s t e p s } = 2 N )$   
8: $\begin{array} { r } { \hat { y } _ { \mathrm { q r y } , i } \gets \arg \operatorname* { m a x } _ { c \in \{ 0 , \dots , C _ { \mathrm { l o s s } } - 1 \} } \sum _ { u = 1 } ^ { H } \sum _ { v = 1 } ^ { W } D _ { \mathrm { q r y } , i } ^ { [ c , u , v ] } \cdot M _ { \mathrm { q r y } , i } ^ { [ u , v ] } \quad \forall i \in \{ 1 , \dots , B _ { \mathrm { q r y } } \} } \end{array}$   
9: $\begin{array} { r } { \mathrm { A c c } ^ { ( 0 ) } \gets \frac { 1 } { B _ { \mathrm { q r y } } } \sum _ { i = 1 } ^ { B _ { \mathrm { q r y } } } \mathbb { I } \left( \hat { y } _ { \mathrm { q r y } , i } = y _ { \mathrm { q r y } , i } \right) \times 1 0 0 \% } \end{array}$   
10: for $k = 1$ to $K _ { \mathrm { r e p e a t s } }$ do   
// 1. Forward Task Inference on Support Set (Active-NCA)   
11: $S _ { \mathrm { s u p p } } \gets \mathrm { e x p a n d } \left( \bar { S } _ { \mathrm { t a s k } } ^ { ( k - 1 ) } , B _ { \mathrm { s u p p } } \right) ^ { \setminus }$   
12: $D _ { \mathrm { s u p p } }  f _ { \theta ^ { * } } ( X _ { \mathrm { s u p p } } , S _ { \mathrm { s u p p } } , \mathrm { s t e p s } = 2 N )$ ▷ Forward unroll without gradients   
$/ / 2 .$ Signed Residual Error Calculation   
13: Construct spatial one-hot targets $Y _ { \mathrm { s p a t i a l } } \in \{ 0 , 1 \} ^ { B _ { \mathrm { s u p p } } \times C _ { \mathrm { l o s s } } \times H \times W }$ from $y _ { \mathrm { s u p p } }$   
14: $E _ { \mathrm { p i x e l } }  ( Y _ { \mathrm { s p a t i a l } } - \mathrm { S o f t m a x } ( D _ { \mathrm { s u p p } } ^ { \mathrm { [ 0 : } C _ { \mathrm { l o s s } } \mathrm { ] } } ) ) \odot M _ { \mathrm { s u p p } }$   
// 3. Cellular Optimization via Error Diffusion $( M e t a – N C A )$   
15: $D _ { \mathrm { m e t a } }  g _ { \phi ^ { * } } ( E _ { \mathrm { p i x e l } } , D _ { \mathrm { m e t a } } , \mathrm { s t e p s } = N )$   
// 4. Cross-Class Spatial Program Superposition   
16: $\begin{array} { r } { S _ { \mathrm { t a s k } } ^ { ( k ) }  \frac { 1 } { B _ { \mathrm { s u p p } } } \sum _ { i = 1 } ^ { \hat { B _ { \mathrm { s u p p } } } } D _ { \mathrm { m e t a } , i } ^ { [ 0 : C _ { s } , : , : ] } } \end{array}$   
$/ / 5 .$ Query Set Evaluation (Gradient-Free Inference)   
17: $S _ { \mathrm { e v a l } } ^ { ( k ) }  \mathrm { r o l l } ( S _ { \mathrm { t a s k } } ^ { ( k ) } , ( \Delta _ { \mathrm { s h i f t } } , \Delta _ { \mathrm { s h i f t } } ) )$ ▷ Optional coordinate shift test   
18: $S _ { \mathrm { q r y } } \gets \mathrm { e x p a n d } \left( S _ { \mathrm { e v a l } } ^ { ( k ) } , B _ { \mathrm { q r y } } \right)$   
19: $D _ { \mathrm { q r y } }  f _ { \theta ^ { * } } ( X _ { \mathrm { q r y } } , S _ { \mathrm { q r y } } , \mathrm { s t e p s } = 2 N )$   
20: $\begin{array} { r } { \hat { y } _ { \mathrm { q r y } , i } \gets \arg \operatorname* { m a x } _ { c } \sum _ { u , v } \left( D _ { \mathrm { q r y } , i } ^ { [ c , u , v ] } \cdot M _ { \mathrm { q r y } , i } ^ { [ u , v ] } \right) \quad \forall i \in \{ 1 , \dots , B _ { \mathrm { q r y } } \} } \end{array}$   
21: $\begin{array} { r } { \mathrm { A c c } ^ { ( k ) } \gets \frac { 1 } { B _ { \mathrm { q r y } } } \sum _ { i = 1 } ^ { B _ { \mathrm { q r y } } } \mathbb { I } \left( \hat { y } _ { \mathrm { q r y } , i } = y _ { \mathrm { q r y } , i } \right) \times 1 0 0 \% } \end{array}$   
22: end for   
23: return Adaptation trajectory $[ \mathsf { A c c } ^ { ( 0 ) } , \mathsf { A c c } ^ { ( 1 ) } , \mathsf { \ldots } , \mathsf { A c c } ^ { ( K _ { \mathrm { r e p e a t s } } ) } ]$ and adapted program $S _ { \mathrm { t a s k } } ^ { ( K _ { \mathrm { r e p e a t s } } ) }$

## C Complete Channel Atlases

![](images/26904fe408a0e82e6f12993b22466c43ddaf4df87f622c97f160baa32f5128a9.jpg)  
Figure 6: Complete atlas of Active-NCA dynamic reasoning channels (Channels 10 through 31) during MNIST inference.

![](images/94548aa57b4777bb5c96bfa73c1c282cca392407eb42d958401317696ca96632.jpg)  
Figure 7: Complete atlas of static spatial program channels $( S _ { \mathrm { t a s k } }$ , Channels 0 through 23) written by the Meta-NCA following support adaptation.

![](images/1561622039a56e9d47b3bc6a16cf39dbcc28835fa987b7f5799c4f1f69153225.jpg)  
(a) Active-NCA Dynamic Reasoning Channels

![](images/c2be92eabbe6d716982e9e8f467f085430504566b9b0fd8604380f9d340b1d43.jpg)  
(b) Meta-NCA Static Program Channels  
Figure 8: Ablation analysis on Omniglot: (a) test accuracy under Gaussian noise injection across dynamic reasoning channels (10 . . . 31), and (b) test accuracy when individual static program channels $\left( 0 \ldots 2 3 \right)$ are ablated to zero.

## D Hidden Channel and Spatial Program Ablation

To understand how task representations and decision-making are distributed across the cellular state channels, we conduct targeted channel ablation experiments on the Omniglot test set.

Active-NCA Dynamic Reasoning Channels (Figure 8a): We inject independent Gaussian noise $( \mathcal { N } ( 0 , 1 ) )$ into individual dynamic reasoning channels $( c \in \{ 1 0 , \ldots , 3 1 \}$ ) throughout test-time execution while keeping all other channels intact. As shown in Figure 8a, the unablated baseline achieves 98.0% accuracy. Ablating individual reasoning channels results in graceful degradation: the majority of channels maintain accuracy above 94%, with only specific channels (e.g., channels 20, 22, and 26) causing moderate drops to $\approx 9 2 \% - 9 4 \%$ . Accuracy is relatively insensitive to individual channel ablation. The higher baseline accuracy compared to the reported main results is due to testing variance, as these results are only tested on one seed.

Meta-NCA Spatial Program Channels (Figure 8b): Similarly, we ablate each of the $C _ { s } = 2 4$ static spatial program channels $( S _ { \mathrm { t a s k } } )$ by setting them to zero following support adaptation. Figure 8b reveals that the baseline (97.0%) drops by at most 1.5%–3.0% when any single program channel is ablated. This demonstrates that the Meta-NCA generates redundant, multi-channel spatial pre-activation codes.

## E Flexible Program

![](images/8edeeb9363df7971137b68346bdf3115b97b48af1509aae04d049198e56e410a.jpg)  
Figure 9: Difference in $S _ { t a s k }$ across 2 independent testing runs on the same training seed

Solution Manifold and Functional Degeneracy. To inspect the stability and uniqueness of the learned spatial programs, we evaluate a single trained model checkpoint (Seed 0) across two independent test-time evaluation runs (Seed 42 vs. Seed 43) on the MNIST benchmark. In this setting, the model parameters $\{ \theta , \phi \}$ remain identical, but the evaluation runs differ across three stochastic dimensions: (i) each run samples an independent 10-shot support set with differing handwriting styles, (ii) the static program grid is initialized from distinct random noise $S _ { \mathrm { t a s k } } ^ { ( 0 ) } \sim \mathcal { U } ( 0 , 1 )$ , and (iii) cellular unrolling is driven by independent stochastic update masks $M _ { \mathrm { s t o c h } } \sim$ Bernoulli(0.5).

As visualized in Figure 9, both runs achieve near identical downstream test accuracy across all 10,000 MNIST test samples (91.92% vs. 91.93%). Subtracting the two adapted programs channel-by-channel $( \Delta S = S _ { \mathrm { t a s k } } ^ { ( \mathrm { S e e d } 4 2 ) } - S _ { \mathrm { t a s k } } ^ { ( \mathrm { S e e d } 4 3 ) } )$ reveals a global Mean Absolute Difference of $\mathbf { M A D } = 0 . 2 0 3 4$ , with localized regions differing by up to ±1.2. This divergence is not random high-frequency static; rather, it manifests as smooth, spatially continuous dipoles and phase shifts. This behavior highlights two core dynamical properties:

Support-Exemplar Adaptation without Overfitting: The localized variations in $S _ { \mathrm { t a s k } }$ reflect local adjustments to the specific stylistic traits (stroke thickness, slant) of the respective support sets, while re maining constrained enough to preserve macro-level class separability across the entire test set.

Functional Degeneracy: The presence of non-trivial spatial divergence alongside identical functional performance demonstrates that the Meta-NCA optimization landscape does not collapse to an isolated, fragile minimum. This Mirrors biological degeneracy, where distinct cellular micro-states reliably execute equivalent computational phenotypes.

## F Simulated spatial-memory corruption

![](images/cbac1b8fddeb97a81f1421b47b25ce83afaedd173dcaf7f882f5b69bb20f8a59.jpg)  
Figure 10: MetaLearnNCA performance on MNIST under different levels of additive random noise to $S _ { t a s k }$

![](images/8c46c64c7e98670eaf32057393cd5c66f6eccee3cfbe4e3e96470ae428f7709d.jpg)  
Figure 11: MetaLearnNCA $S _ { t a s k }$ under different levels of noise.

Visualizing Additive Noise Robustness. To provide intuitive grounding for the noise resilience curve in Figure 10, Figure 11 compares the clean spatial program grid against a grid corrupted by additive Gaussian noise (σ = 0.5) across 100% of spatial coordinates. As visible in the bottom row, this perturbation severely degrades local pixel fidelity, simulating high-amplitude thermal or transmission noise in physical analog substrates. Crucially, the global low-frequency modes of the program remain somewhat structurally coherent.

## G Dataset to Dataset Transfer

![](images/d4ff51cbff2a6a96ae63b4fc49bd4b3c8436c12fac451853a58f6688d541c41f.jpg)  
Figure 12: Cross dataset transfer, each model is trained on one dataset and tested on all the other ones. 50 repeats per dataset with 95% CI intervals.

# H K-repeats and K-shot meta-learning surface

MetaNCA Few-Shot Generalization Landscape: Accuracy vs. Support Shots & Optimization Repeats (k) (Evaluated using Omniglot-Trained Dual-NCA Model)

![](images/a75dcfdfa95dc265f8269c1c62d0fc5d4b6432e3ec7eee0306cdd9990cc79d17.jpg)  
Figure 13: Dataset surfaces for seed 0, trained on OMNIGLOT across a combination of K-repeats and Kshots.

## I Spatial Translation and Coordinate Invariance Robustness

A core design characteristic of METALEARNNCA is that task knowledge is encoded as a continuous 2D spatial field $( S _ { \mathrm { t a s k } } \in \mathbb { R } ^ { C _ { s } \times H \times W } )$ rather than a global 1D latent vector. While this inductive bias provides localized routing mechanisms for stroke- and shape-based reasoning, it introduces a natural question: does the coupled cellular dynamics rely on rigid pixel-to-pixel coordinate alignment, or does it exhibit translational elasticity under spatial displacement?

## I.1 Experimental Setup

To quantify spatial coordinate robustness, we evaluate a model meta-trained on Omniglot (Seed 0) in a standard 5-shot, 10-way evaluation setting across all six benchmark domains. Following support-set adaptation on canonical images, we apply independent discrete 2D spatial translations to all query instances:

$$
X _ { \mathrm { s h i f t e d } } ( u , v ) = X ( u + \Delta _ { u } , v + \Delta _ { v } ) ,\tag{7}
$$

with displacement thresholds $\Delta \in \{ 1 , 2 , 3 , 4 \}$ pixels, using circular padding to preserve canvas dimensions $( H = W = 2 8 )$ . We evaluate classification accuracy across 5 independent evaluation runs per displacement setting, and report sample means with 95% confidence intervals.

## I.2 Results and Dynamical Elasticity

As summarized in Table 2, METALEARNNCA displays graceful, non-catastrophic degradation across all evaluated visual domains:

Table 2: Robustness of METALEARNNCA to discrete spatial coordinate shifts $( \Delta \in \{ 1 , 2 , 3 , 4 \}$ pixels) in the 5-shot, 10-way transfer regime. Results report mean classification accuracy $\pm 9 5 \%$ confidence intervals evaluated over 5 independent testing runs on full test splits.
<table><tr><td>Dataset</td><td> $\mathbf { S h i f t = 1 \ p x }$ </td><td> $\mathbf { S h i f t } = 2 \mathbf { \ p x }$ </td><td> $\mathbf { S h i f t } = 3 \mathbf { \ p x }$ </td><td> $\mathbf { S h i f t } = 4 \mathbf { \ p x }$ </td></tr><tr><td>Omniglot (In-Dist)</td><td> $9 3 . 4 7 \pm 4 . 5 1 \%$ </td><td> $9 0 . 9 3 \pm 4 . 5 6 \%$ </td><td> $8 3 . 3 3 \pm 4 . 8 6 \%$ </td><td> $7 0 . 1 3 \pm 8 . 8 1 \%$ </td></tr><tr><td>MNIST</td><td> $8 4 . 4 3 \pm 1 . 4 1 \%$ </td><td> $8 2 . 9 3 \pm 2 . 6 2 \%$ </td><td> $7 7 . 1 3 \pm 2 . 8 6 \%$ </td><td> $5 9 . 0 1 \pm 3 . 3 7 \%$ </td></tr><tr><td>USPS</td><td> $5 7 . 8 9 \pm 7 . 2 7 \%$ </td><td> $5 1 . 6 7 \pm 6 . 8 5 \%$ </td><td> $4 0 . 7 1 \pm 5 . 3 1 \%$ </td><td> $3 0 . 8 3 \pm 2 . 5 1 \%$ </td></tr><tr><td>Fashion-MNIST</td><td> $4 8 . 7 2 \pm 4 . 5 6 \%$ </td><td> $4 2 . 8 5 \pm 4 . 0 1 \%$ </td><td> $3 4 . 0 3 \pm 5 . 1 6 \%$ </td><td> $2 4 . 7 2 \pm 4 . 6 6 \%$ </td></tr><tr><td>KMNIST</td><td> $4 1 . 7 0 \pm 3 . 8 8 \%$ </td><td> $3 7 . 4 7 \pm 4 . 0 2 \%$ </td><td> $3 0 . 5 0 \pm 3 . 5 5 \%$ </td><td> $2 2 . 7 3 \pm 3 . 7 5 \%$ </td></tr><tr><td>MedMNIST</td><td> $1 8 . 5 5 \pm 8 . 0 5 \%$ </td><td> $1 7 . 2 3 \pm 7 . 9 2 \%$ </td><td> $1 4 . 9 8 \pm 6 . 9 8 \%$ </td><td> $1 3 . 0 0 \pm 5 . 9 0 \%$ </td></tr></table>

1. Near-Invariance at Local Offsets $( \Delta \leq 2 { \bf p x } ) \colon$ For local shifts of $\Delta = 1$ pixel, accuracy remains virtually indistinguishable from clean unshifted baselines (e.g., MNIST achieves $8 4 . 4 3 \% \pm 1 . 4 1 \% ,$ and Omniglot attains $9 3 . 4 7 \% \pm 4 . 5 1 \% )$ . $\mathrm { { A t } } \Delta = 2$ pixels, degradation remains minimal, with MNIST dropping by only 1.50% (82.93%) when compared to $\Delta = 1$ and in-distribution Omniglot retaining $> 9 0 . 9 \%$ mean accuracy.

2. Degradation under Larger Offsets $( \Delta = 4 { \bf p } { \bf x } ) \colon$ : When images are perturbed by $\Delta = 4$ pixels, representing a larger displacement exceeding 14.2% of the total canvas width $( 2 8 \times 2 8 )$ , the model retains some task structure. On MNIST, the model achieves $5 9 . 0 1 \% \pm 3 . 3 7 \%$ , which is nearly $6 \times$ higher than the 10.0% random-guess baseline. Similarly, Omniglot retains $7 0 . 1 3 \% \pm 8 . 8 1 \%$

Mechanistic Basis of Elasticity. This shows that $S _ { \mathrm { t a s k } }$ does not solely function as an overfitted, static pixel mask. Rather, some degree of translational elasticity is an emergent consequence of the Active-NCA’s recurrent message-passing dynamics. Because each update step convolves differential operators $( K _ { \mathrm { s o b e l } } , K _ { \mathrm { l a p } } )$ across a 3 × 3 neighborhood, information propagates outwards at 1 pixel per step. Across forward inference steps, each cell integrates context over an effective receptive field spanning the entire canvas. This recurrent diffusion acts as an intrinsic spatial regularizer: local gradients dynamically warp and re-route displaced stroke trajectories into the attractor basin defined by $S _ { \mathrm { t a s k } }$ , buffering the system against some coordinate misalignment.

## J Baseline Hyperparameters and Dataset Preprocessing

All canonical baselines share a standard Conv4-64 encoder (4 conv blocks with 64 channels, ReLU, $2 \times 2$ max-pooling, and global average pooling yielding a 64D feature vector). All models are meta-trained for 5,000 epochs on Omniglot with AdamW $( \mathrm { l r } = 1 0 ^ { - 3 }$ , weight decay $= 1 0 ^ { - 5 }$ , gradient clipping at 1.0). Full source code and configuration files are available in our anonymous repository.<sup>1</sup>

Table 3: Architectural and optimization hyperparameters for canonical baselines.
<table><tr><td>Hyperparameter</td><td>ProtoNet</td><td>MatchingNet</td><td>RelationNet</td><td>FOMAML</td></tr><tr><td>Backbone</td><td>Conv4-64</td><td>Conv4-64</td><td>Conv4-64</td><td>Conv4-64</td></tr><tr><td>Parameters</td><td>≈115k</td><td>≈115k</td><td>≈ 225k</td><td>≈115k</td></tr><tr><td>Output Head</td><td>Metric Centroid</td><td>Cosine Attention</td><td>Relation MLP</td><td>Linear (64 → 10)</td></tr><tr><td>BatchNorm Stats</td><td>Frozen Running</td><td>Frozen Running</td><td>Frozen Running</td><td>Shuffled Mini-Batch</td></tr><tr><td>Meta-Loss</td><td>Cross-Entropy</td><td>Neg. Log-Likelihood</td><td>Mean Squared Error</td><td>Cross-Entropy</td></tr><tr><td>Inner Steps  $( K _ { \mathrm { t r a i n } } / K _ { \mathrm { t e s t } } )$ </td><td></td><td></td><td></td><td>3/5</td></tr><tr><td>Inner LR  $( \alpha _ { \mathrm { t r a i n } } )$ </td><td></td><td></td><td></td><td>0.1 (clip = 1.0)</td></tr><tr><td>In-Domain Inner  $\mathrm { L R } \left( \alpha _ { \mathrm { o m n i } } \right)$ </td><td></td><td></td><td></td><td>0.2 (≤ 5-shot), 0.1 (10-shot)</td></tr><tr><td>OOD Inner LR  $( \alpha _ { \mathrm { o o d } } )$ </td><td></td><td></td><td></td><td>0.1 (1s), 0.05 (5s), 0.03 (10s)</td></tr><tr><td>Higher-Order Gradients</td><td></td><td></td><td></td><td>First-Order (FOMAML)</td></tr></table>

Table 4: Dataset preprocessing and normalization pipelines $( 2 8 \times 2 8 $ grayscale).
<table><tr><td>Dataset</td><td>Domain</td><td>Transform Pipeline</td></tr><tr><td>Omniglot</td><td>Sparse Characters (In-Dist)</td><td>Resize(28) → ToTensor() → Invert (1.0 − x) → Normalize(0.5, 0.5)</td></tr><tr><td>MNIST/KMNIST/USPS</td><td>Digits / Calligraphy (OOD)</td><td>Resize(28) → ToTensor() → Normalize(0.5, 0.5)</td></tr><tr><td>Fashion / MedMNIST</td><td>Silhouettes / CT Organs (OOD)</td><td>Resize(28)() → ToTensor() → Normalize(0.5, 0.5)</td></tr></table>