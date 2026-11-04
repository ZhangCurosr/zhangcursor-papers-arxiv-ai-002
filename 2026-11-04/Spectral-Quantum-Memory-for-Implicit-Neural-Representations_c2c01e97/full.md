# Spectral Quantum Memory for Implicit Neural Representations

Daniel A. Moroz-Broitman Department of Computer Science Ben Gurion University morozd@post.bgu.ac.il

Eliya Nachmani Department of Electrical Engineering Ben Gurion University eliyanac@bgu.ac.il

## Abstract

Implicit neural representations (INRs) model signals as continuous functions from coordinates to values, but their performance depends strongly on how spectral capacity is allocated. Yet most Quantum INR (QINR) architectures treat spectral components as independent outputs of separate quantum blocks. We introduce the Spectral Quantum Memory Model (SQMM), a memory coupled quantum INR in which sequential hybrid data reuploading layers (bands) communicate through a shared coherent quantum memory register. Each band reads the memory, computes on its own coordinate upload, and writes part of its state back, so later bands can build on the computation of earlier ones. When memory read and write layers are input independent, we prove that the memory state forms an operator valued Fourier series whose support is contained in the Minkowski sum of the initial memory spectrum and the local band spectra. Downstream bands then condition their Fourier coefficients on the spectral content stored by earlier bands, and the memory coupling creates no new primitive frequencies. This theory motivates a spectral memory alignment loss that encourages memory to encode the spectrum expressed by the deployed prefix of the model. Across audio and image reconstruction benchmarks, SQMM improves over compact QINR and INR baselines. Code is provided in: https://github.com/Kiiwii10/SQMM

## 1 Introduction

Implicit neural representations (INRs) model signals as continuous coordinate to value functions, enabling compact representations for images, audio (Figure 1), video, geometry, radiance fields, and physical signals [81, 51, 46, 67, 14, 70]. Their main bottleneck is spectral. Standard MLPs learn low frequency structure before high frequency detail [54, 82], so INR performance depends on how frequency content is exposed, selected, and composed [6]. This has motivated Fourier features [70], periodic activations [67], multiplicative filters [16], band limited and multiscale coordinate networks [39, 61], wavelet INRs [57], and adaptive periodic bases [40, 44]. These works show that spectral organization is central to INR capacity.

Quantum implicit neural representations (QINRs) make this spectral structure explicit because data reuploading circuits compute finite Fourier series. Data reuploading repeatedly injects classical inputs into variational quantum circuits [47, 53, 22]. Their accessible frequencies are determined by encoding gates, while trainable circuit parameters determine the coefficients [60, 9, 83, 79, 3, 45]. QIREN formulates QINRs as quantum generalizations of Fourier neural networks [85], and QVF extends QINRs toward quantum centric visual fields through neural amplitude encoding and projective measurement [77]. Quantum recurrent models and quantum LSTMs use hidden quantum states or variational cells as memory mechanisms [5, 12]. Existing QINRs, however, do not characterize how memory qubits share spectral information across coordinate frequency modules.

![](images/71229f20bb1951db168458cccd80df67ea4bed19b758ba452f05f6d38a89ef22.jpg)  
Figure 1: Sound representation results adapted from QIREN [85], showing normalized amplitude over normalized time.

In existing multiband QINRs, the bands operate largely independently, and what one band computes reaches the others only through the classical decoder. A shared memory lets information produced by earlier bands influence later ones. This paper introduces the Spectral Quantum Memory Model (SQMM), a memory coupled QINR built from sequential data reuploading circuits organized into hybrid quantum and classical bands. Each band reads a shared unmeasured memory register, performs local spectral computation, and writes back before the next band. SQMM therefore propagates coherent spectral context across bands, allowing later bands to condition coefficient allocation on earlier spectral computation instead of relearning each band independently.

When memory read and write layers are independent of the input coordinate, each memory coupled band remains in the data reuploading model class. The reduced memory state is an operator valued Fourier series whose support is contained in the Minkowski sum of the initial memory spectrum and the band spectra. Downstream readout coefficients arise from convolutions between the memory spectrum and local band spectra. Thus memory does not introduce arbitrary primitive frequencies. It preserves the Fourier properties of QINRs while expanding spectral context between bands.

This view motivates a spectral memory alignment loss. The loss aligns memory with an estimate of the accumulated spectral occupancy of the deployed prefix contribution. The supervised loss trains the predictor, while the alignment loss calibrates memory as a readable spectral summary of the computation so far. Downstream bands can still reuse, refine, or correct upstream frequencies.

The contributions are fourfold. First, SQMM introduces a memory coupled QINR that propagates coherent spectral context across sequential data reuploading bands. Second, the theory proves that memory preserves the Fourier properties of data reuploading circuits, since the memory state remains an operator valued Fourier series whose support is contained in the Minkowski sum of the band spectra. Third, a spectral memory alignment loss trains memory to summarize the deployed prefix spectrum. Fourth, SQMM improves on the reconstruction results of quantum and classical INR baselines on audio and image representation tasks under small parameter budgets.

## 2 Related Work

Implicit neural representations and spectral coordinate networks. Coordinate based neural repre sentations originate in compositional pattern producing networks [69] and now support continuous representation of signals, shapes, scenes, and fields [81], including shape templates [21], continuous shape distributions [86], signed distance functions [51], scene representation networks [66], local implicit grids [29], neural radiance fields [46], image generation and compression [68, 2, 14, 10, 49], time indexed modeling [80], and scientific modeling [28]. Their central limitation is spectral bias, since standard networks tend to fit low frequencies before high frequency detail [54, 82, 4, 70]. This has motivated spectral INR designs based on Fourier features [70], sinusoidal and nonstandard activations [50, 67, 55], multiplicative filters [16], band limited and multiscale networks [39, 61], wavelet INRs [57], adaptive periodic bases [40, 44], Fourier reparameterized training [62], and, more recently, content aware Fourier Chebyshev encodings (CAFE) [30], multi scale sine activations (MSA-INR) [24], frequency and locality aware representations (FLAIR) [33], robust frequency tuning for sinusoidal networks (TUNER) [48], variance informed initialization (VI3NR) [34], and inductive gradient adjustment against spectral bias (IGA) [63]. These methods shape the spectrum of a single classical network through its encoding, activation, initialization, or gradient, and none of them passes a state between separately computed frequency modules. SQMM follows this spectral view, but studies spectral sharing in quantum data reuploading circuits where the finite Fourier structure can be analyzed explicitly.

Parameterized quantum models and data reuploading. Parameterized quantum circuits are trainable models for supervised learning, quantum feature maps, generative modeling, reinforcement learning, and hybrid quantum classical computation [43, 47, 15, 13, 25, 7, 65, 58, 8]. Hybrid quantum neural networks combine classical processing with quantum circuits that implement trainable feature spaces, kernels, or intermediate representations [59, 23], with applications in sequence modeling, classification, reinforcement learning, generative modeling, and architecture search [12, 38, 11, 42, 20, 76]. Data reuploading repeatedly injects classical variables into variational quantum circuits and improves expressivity even with small registers [53, 22]. Angle encoded quantum models compute finite Fourier series whose accessible frequencies are determined by encoding generator spectra, while trainable parameters determine the coefficients [60]. This Fourier view has been extended to single qubit native models [83], multidimensional Fourier series [9], universal approximation in quantum enhanced feature spaces [23], structured and exponential encodings [64, 75], exact realized spectra and coefficients [79], coefficient constraints from encoding redundancy [45], and gradient profiles in deep reuploading circuits [3]. SQMM extends this Fourier framework from isolated reuploading circuits to stacked memory coupled bands.

Quantum implicit neural representations. The closest prior work is QIREN, which connects data reuploading circuits to INRs by treating parameterized quantum circuits as quantum analogues of Fourier feature networks [85]. QIREN uses repeated coordinate encodings to produce quantum Fourier features decoded by classical layers for signal representation, image super resolution, and image generation. QVF moves toward quantum centric visual fields using neural amplitude encoding and projective measurement over entangled ansatzes for 2D and 3D signals [77]. SQMM is comple mentary to both models. It retains the data reuploading Fourier view of QIREN but couples its bands through a shared coherent memory, and it differs from QVF by emphasizing spectral communication across reuploading bands rather than amplitude encoding or direct visual field measurement. Prior Fourier analyses of data reuploading circuits characterize circuits with fixed incoming quantum states [60, 83, 9]. QIREN connects that theory to INRs, but it measures each quantum block and passes a classical feature vector to the next hybrid layer, so its analysis characterizes the Fourier form of an individual circuit in its immediate input. SQMM instead keeps a coherent memory state across sequential bands, and our analysis describes how the operator valued spectrum of that state conditions downstream quantum features throughout the model.

Memory, recurrence, and auxiliary spectral supervision. Memory and recurrence have appeared in quantum learning through recurrent quantum neural networks, quantum LSTM style architectures, and quantum reservoir computing [5, 12, 19]. Classical models also use inter stage coupling through residual or stagewise refinement, where later modules correct earlier computations [26, 18]. In INRs, band limited and multiscale coordinate networks explicitly organize frequency bands [39, 61], while spectral regularization and auxiliary supervision shape learned representations during training [62, 37]. Quantum recurrent models use hidden quantum states as memory, but they do not characterize the coordinate frequency content of that state or its effect on downstream QINR coefficients. SQMM differs from these classical mechanisms because the coupled state is not a residual stream, recurrent hidden vector, or auxiliary prediction head. It is a quantum density matrix with operator valued Fourier coefficients that remains readable by downstream quantum bands.

## 3 Background

We consider coordinate based signal representation. Coordinates are x $\mathbf { \Lambda } \in \mathbb { R } ^ { d } .$ , targets are $\boldsymbol { y } \in \mathbb { R } ^ { d _ { y } }$ and the model learns a continuous map $x \mapsto y$ . SQMM first maps x to a band specific uploaded variable

$$
\gamma _ { i } ( x ) = A _ { i } \operatorname { P E } ( x ) + b _ { i } \in \mathbb { R } ^ { g _ { i } } , \qquad A _ { i } \in \mathbb { R } ^ { g _ { i } \times p } , \qquad b _ { i } \in \mathbb { R } ^ { g _ { i } } .\tag{1}
$$

Here $\mathrm { P E } : \mathbb { R } ^ { d }  \mathbb { R } ^ { p }$ is a fixed positional or Fourier feature encoding [73, 70, 46]. SQMM uses quantum data reuploading circuits inside this coordinate model. A register is a group of qubits treated as one subsystem. Band i has a local data register $D _ { i }$ with $q _ { i }$ qubits, and all bands share a memory register M with m qubits. The fixed new data state is $\eta _ { D _ { i } }$ , and the memory state after band i is a density matrix $\sigma _ { M } ^ { ( i ) } ( x )$ , namely a positive semidefinite matrix with trace one. A quantum operation is written as a channel. The band channel $B _ { i } ( \gamma _ { i } ( x ) )$ acts on the joint register $D _ { i } \otimes M$ . Table 7 in Appendix 8 collects the notation.

![](images/75befd03d339854fffcf00472bb55c6cb9f488aba32eb9a5991042fc5a4dd865.jpg)

Quantum Memory Qubits  
![](images/dbd6bd07f0905603f0a2044a9aae32f38d3f3025f825c1e3cb7163957952c936.jpg)  
Figure 2: SQMM architecture. Coordinates are mapped to band uploads $\gamma _ { i } ( x )$ and to the memory upload $\gamma _ { M } ( \boldsymbol { x } )$ , both of the form of Eq. (1). Sequential quantum bands share the memory register M and emit Pauli expectation features, which a classical decoder combines. Dashed red arrows mark the training only spectral alignment loss. Numbers follow the Intuition paragraph.

We use $\operatorname { T r } [ \cdot ]$ for trace and $\mathrm { T r } _ { D _ { i } } [ \cdot ]$ for partial trace over the local data register. An observable O is a Hermitian matrix that represents a measurable quantity, and $\operatorname { T r } [ O \rho ]$ is its expectation value in state $\rho .$ In SQMM these expectations are used as classical features.

A finite data reuploading circuit has a finite Fourier expansion in the uploaded variable because input dependent gates are generated by finite dimensional Hermitian operators [60, 9, 83]. For a generic uploaded variable $z \in \mathbb { R } ^ { g _ { i } }$ , the local upload support of band i is denoted by $\Omega _ { i , \gamma } ^ { \mathrm { l o c } }$ , with terms $e ^ { \mathrm { i } \langle \omega _ { \gamma } , z \rangle }$ for $\omega _ { \gamma } \in \Omega _ { i , \gamma } ^ { \mathrm { l o c } }$ . After substituting $z = \gamma _ { i } ( x )$ , the analysis uses the induced coordinate support $\Omega _ { i , x } ^ { \mathrm { l o c } } .$ . This support is exact when the upload is affine in the raw coordinate. With fixed sinusoidal positional encodings or nonlinear angle maps, as commonly used in coordinate networks and QINRs [70, 46, 85], the composed coordinate dependence is analyzed either in the lifted encoding coordinates or through the finite grid projection used by the spectral alignment loss. Thus the theory applies to an exact finite coordinate expansion when available, and otherwise to the chosen finite truncation. For sets $U , V \subset \mathbb { R } ^ { d }$ , let $U ^ { \cdot } + V = \{ u + v \mid u \in U , v \in V \}$ denote their Minkowski sum. Since memory is a density matrix, its Fourier coefficients are matrices rather than scalar hidden features. Additional quantum notation and the upload to coordinate support conversion are given in Appendix 9.

## 4 Spectral Quantum Memory Model

SQMM (Figure 2) is a hybrid quantum and classical INR for learning $x \mapsto y$ . It consists of n sequential data reuploading bands connected by a shared memory register. Each band emits expectation valued classical features, while the memory state is updated and passed to later bands. The memory read and write layers are input independent, so they change Fourier coefficients but do not create new primitive coordinate frequencies.

Intuition. Without memory, each band of a multiband QINR sees only the coordinate, and its features meet those of other bands only in the classical decoder. SQMM adds a small register of m unmeasured memory qubits that is carried from band to band. For a coordinate x, the memory is first prepared from the coordinate by a one time upload $\gamma _ { M } ( x ) \left( 1 \right)$ . Each band then reads the memory through input independent couplings (2), uploads its own coordinate encoding $\gamma _ { i } ( x )$ several times (3), and writes part of its local state back into memory (4) before its local qubits are discarded. The band emits Pauli expectation features, and the updated memory is passed to the next band (5). (1)-(5) refer to marks in Figure 2. At inference only the band features are estimated, and the memory is never measured. Section 4.2 shows what this pathway can carry, and Section 4.3 introduces a loss that trains the memory to resemble the spectrum accumulated by the preceding bands.

## 4.1 Architecture

Band i starts with a new data state $\eta _ { D _ { i } }$ and incoming memory $\sigma _ { M } ^ { ( i - 1 ) } ( x )$ . Its uploaded variable is $\gamma _ { i } ( x )$ from Eq. (1). The band channel is

$$
\begin{array} { r } { \mathcal { B } _ { i } ( \gamma _ { i } ( x ) ) = \mathcal { V } _ { i , L _ { i } + 1 } \circ S _ { i , L _ { i } } ( \gamma _ { i } ( x ) ) \circ \mathcal { V } _ { i , L _ { i } } \circ \cdots \circ S _ { i , 1 } ( \gamma _ { i } ( x ) ) \circ \mathcal { V } _ { i , 1 } . } \end{array}\tag{2}
$$

The layers $\mathcal { S } _ { i , \ell }$ are the only input dependent layers. The layers $\nu _ { i , \ell }$ are trainable and input independent, and include local gates, entangling gates, memory read couplings, memory write couplings, and memory only processing. In the deployed circuit, $\mathcal { V } _ { i , 1 }$ holds the memory read couplings, each upload is followed by a local entangling layer, and the closing block $\mathcal { V } _ { i , L _ { i } + 1 }$ holds the last local entangling layer together with the memory write couplings. Therefore memory coupling preserves the data reuploading form on $D _ { i } \otimes M$ . The joint post band state is

$$
\xi _ { i } ( x ) = { \cal B } _ { i } ( \gamma _ { i } ( x ) ) \left( \eta _ { D _ { i } } \otimes \sigma _ { M } ^ { ( i - 1 ) } ( x ) \right) .\tag{3}
$$

The updated memory is obtained by discarding the local data register,

$$
\sigma _ { M } ^ { ( i ) } ( x ) = \mathrm { T r } _ { D _ { i } } \left[ \xi _ { i } ( x ) \right] .\tag{4}
$$

Band i emits Pauli expectation features

$$
\pmb { f } _ { i } ( \pmb { x } ) = \left( \mathrm { T r } [ ( O _ { i , 1 } \otimes I _ { M } ) \xi _ { i } ( \pmb { x } ) ] , \ldots , \mathrm { T r } [ ( O _ { i , d _ { f , i } } \otimes I _ { M } ) \xi _ { i } ( \pmb { x } ) ] \right) \in \mathbb { R } ^ { d _ { f , i } } .\tag{5}
$$

These features are ensemble expectations. The downstream memory is not conditioned on local measurement outcomes. Hardware estimation uses repeated circuit executions, and the readout semantics are formalized in Appendix 11. Let $\begin{array} { r } { d _ { F } = \sum _ { i = 1 } ^ { \hat { n } } d _ { f , i } } \end{array}$ , let

$$
\pmb { f } ( \pmb { x } ) = [ \pmb { f } _ { 1 } ( \pmb { x } ) ^ { \top } , \ldots , \pmb { f } _ { n } ( \pmb { x } ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { d _ { F } } ,\tag{6}
$$

and let $G _ { \theta } : \mathbb { R } ^ { d _ { F } }  \mathbb { R } ^ { d _ { y } }$ be a classical decoder. The prediction is

$$
{ \hat { y } } ( x ) = G _ { \theta } ( f ( x ) ) .\tag{7}
$$

## 4.2 Spectral Interpretation

A data reuploading circuit computes a finite Fourier series whose frequencies are fixed by its encoding gates. The result below states that this remains true for a band that receives a coordinate dependent memory state. Every frequency that leaves a band, in its features or in the memory passed on, is the sum of a frequency already present in the incoming memory and a frequency introduced by the band’s own uploads. The input independent layers of the band decide how strongly each such frequency appears. The memory is therefore a channel through which later bands can reuse and reweight the spectral content of earlier bands.

The spectral analysis expands each band on the coordinate support $\Omega _ { i , x } ^ { \mathrm { l o c } }$ . For a fixed band i, assume the incoming memory has the operator valued Fourier expansion

$$
\sigma _ { M } ^ { ( i - 1 ) } ( x ) = \sum _ { \mu \in \Omega _ { M , x } ^ { ( i - 1 ) } } e ^ { \mathrm { i } \left. \mu , x \right. } \tau _ { \mu } ^ { ( i - 1 ) } , \qquad \tau _ { \mu } ^ { ( i - 1 ) } \in \mathbb { C } ^ { 2 ^ { m } \times 2 ^ { m } } .\tag{8}
$$

Here $\tau _ { \mu } ^ { ( i - 1 ) }$ is the matrix coefficient of the memory state at coordinate frequency µ. Write the coordinate expansion of the band channel as

$$
B _ { i } ( \gamma _ { i } ( x ) ) = \sum _ { \nu \in \Omega _ { i , x } ^ { \mathrm { l o c } } } e ^ { \mathrm { i } \left. \nu , x \right. } B _ { i , \nu } .\tag{9}
$$

Lemma 1 (Spectral propagation through memory). Under Eqs. (8) and (9), the outgoing memory and emittedfeatures ofband i have no coordinatefrequencies outside $\Omega _ { M , x } ^ { ( i - 1 ) } + \Omega _ { i , x } ^ { \mathrm { l o c } }$ . In particular,

$$
\Omega _ { M , x } ^ { ( i ) } \subseteq \Omega _ { M , x } ^ { ( i - 1 ) } + \Omega _ { i , x } ^ { \mathrm { l o c } } , \qquad \Omega _ { f _ { i } , x } \subseteq \Omega _ { M , x } ^ { ( i - 1 ) } + \Omega _ { i , x } ^ { \mathrm { l o c } } .\tag{10}
$$

Input independent read, write, and memory only layers can change Fourier coefficients but cannot enlarge these supports (Proposition 1 and Lemma 2).

Proof. The main point is the product rule for Fourier phases. Substituting Eqs. (8) and (9) into Eq. (4) gives

$$
\sigma _ { M } ^ { ( i ) } ( x ) = \sum _ { \omega } e ^ { \mathrm { i } \left. \omega , x \right. } \tau _ { \omega } ^ { ( i ) } , \qquad \tau _ { \omega } ^ { ( i ) } = \sum _ { \mu + \nu = \omega } \mathrm { T r } _ { D _ { i } } \left[ \mathcal { B } _ { i , \nu } \left( \eta _ { D _ { i } } \otimes \tau _ { \mu } ^ { ( i - 1 ) } \right) \right] .\tag{11}
$$

Thus an outgoing memory frequency ω can appear only when $\omega = \mu + \nu _ { : }$ , where $\mu$ is already present in memory and ν is introduced by the current band. The same argument gives the emitted feature spectrum. For feature component a, define

$$
\ell _ { i , a , \nu } ( X ) = \operatorname { T r } \left[ ( O _ { i , a } \otimes I _ { M } ) B _ { i , \nu } ( \eta _ { D _ { i } } \otimes X ) \right] .\tag{12}
$$

This map is linear in the memory side matrix X. Applying it to Eq. (8) gives

$$
f _ { i , a } ( x ) = \sum _ { \omega \in \Omega _ { M , x } ^ { ( i - 1 ) } + \Omega _ { i , x } ^ { \mathrm { l o c } } } c _ { \omega , a } ^ { ( i ) } e ^ { \mathrm { i } \left. \omega , x \right. } , \qquad c _ { \omega , a } ^ { ( i ) } = \sum _ { \mu + \nu = \omega } \ell _ { i , a , \nu } \left( \tau _ { \mu } ^ { ( i - 1 ) } \right) .\tag{13}
$$

This proves the support claim.

The lemma shows that memory does not introduce arbitrary primitive frequencies. It lets downstream bands allocate coefficients conditional on an operator valued spectral summary of upstream computation. The one time memory upload $\gamma _ { M } ( \boldsymbol { x } )$ before the first band (Figure 2) gives the base support $\Omega _ { M , x } ^ { ( 0 ) }$ , which is {0} when memory is initialized independently of x. Applying Eq. (10) recursively bounds the memory support after band i by $\begin{array} { r } { \Omega _ { M , x } ^ { ( 0 ) } + \sum _ { j = 1 } ^ { i } \Omega _ { j , x } ^ { \mathrm { l o c } } } \end{array}$ (Proposition 2 and Corollary 2 in Appendix 10).

## 4.3 Spectral Memory Alignment Loss

Lemma 1 says that memory can carry a spectral summary of the prefix computation. The alignment loss makes this summary readable. It compares two spectra on a finite grid,

$$
\Omega _ { \mathrm { g r i d } } = \{ \bar { \omega } _ { 1 } , \dots , \bar { \omega } _ { N _ { \omega } } \} \subset \mathbb { R } ^ { d } .\tag{14}
$$

Bars denote grid frequencies. Hats denote coefficients estimated from sampled coordinates.

Feature spectrum. For sampled coordinates $x _ { 1 } , \dots , x _ { T }$ , define

$$
\Phi _ { t h } = e ^ {  { \mathrm { i } } \left. \bar { \omega } _ { h } , x _ { t } \right. } , \qquad \widehat { a } ( u ) = ( \Phi ^ { \dagger } \Phi + \varepsilon I _ { N _ { \omega } } ) ^ { - 1 } \Phi ^ { \dagger } u .\tag{15}
$$

All signals in this paper are sampled on regular grids. We therefore take $\Omega _ { \mathrm { g r i d } }$ from the discrete Fourier frequencies of the sampling grid and set $\varepsilon = 0$ , so Eq. (15) reduces to the discrete Fourier transform scaled by $1 / T$ and is computed with an FFT (Appendix 12). Applying this estimator to each feature component of band j gives $\widehat { \pmb { c } } _ { j , h } ^ { F } = ( \widehat { \pmb { c } } _ { j , 1 , h } ^ { F } , \ldots , \widehat { \pmb { c } } _ { j , d _ { f , j } , h } ^ { F } ) \in \mathbb { C } ^ { d _ { f , j } ^ { \star } }$ , and the prefix feature occupancy is

$$
P _ { i } ^ { F } ( \bar { \omega } _ { h } ) = \sum _ { j = 1 } ^ { i } \| \widehat { \pmb { c } } _ { j , h } ^ { F } \| _ { 2 } ^ { 2 } .\tag{16}
$$

This records which grid frequencies have been expressed by the emitted features of bands $1 , \ldots , i ,$ before the final decoder.

Memory spectrum. The memory is probed with the unnormalized m qubit Pauli string basis $\mathcal { Q } _ { m } = \dot { \{ I , X , Y , Z \} } ^ { \otimes m }$ , with $| \mathcal { Q } _ { m } | = 4 ^ { \bar { m } }$ and $\mathrm { T r } ( Q _ { s } Q _ { s ^ { \prime } } ) = 2 ^ { m } { \bf 1 } \{ s = s ^ { \prime } \}$ . For each $Q _ { s } \in \mathcal { Q } _ { m }$ , the

scalar memory probe is $\chi _ { i , s } ^ { M } ( x ) = \mathrm { T r } [ Q _ { s } \sigma _ { M } ^ { ( i ) } ( x ) ]$ . Estimating its Fourier coefficients with Eq. (15) and collecting them at $\bar { \omega } _ { h }$ gives $\widehat { \pmb { c } } _ { i , h } ^ { M } = ( \widehat { \pmb { c } } _ { i , s , h } ^ { M } ) _ { { \pmb { Q } } _ { s } \in { \pmb { Q } } _ { m } } \in \mathbb { C } ^ { 4 ^ { m } }$ . The memory occupancy spectrum is

$$
P _ { i } ^ { M } ( \bar { \omega } _ { h } ) = 2 ^ { - m } \sum _ { Q _ { s } \in \mathcal { Q } _ { m } } | \widehat { c } _ { i , s , h } ^ { M } | ^ { 2 } .\tag{17}
$$

The factor $2 ^ { - m }$ follows from the Pauli trace normalization and cancels under the normalization of Eq. (18) up to a rescaling of δ. If the finite expansion is exact and the full Pauli basis is used, then $P _ { i } ^ { M } ( \bar { \omega } _ { h } ) = \| \tau _ { \bar { \omega } _ { h } } ^ { ( i ) } \| _ { F } ^ { 2 }$ up to finite grid estimation error. The derivation is given in Appendix 12.

Alignment objective. For any nonnegative spectrum $R : \Omega _ { \mathrm { g r i d } }  \mathbb { R } _ { \geq 0 }$ , define

$$
\mathcal { N } _ { \delta } R ( \bar { \omega } _ { h } ) = \frac { R ( \bar { \omega } _ { h } ) + \delta } { \sum _ { g = 1 } ^ { N _ { \omega } } R ( \bar { \omega } _ { g } ) + N _ { \omega } \delta } .\tag{18}
$$

With stop gradient sg, the alignment loss is

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \sum _ { i = 1 } ^ { n - 1 } \sum _ { h = 1 } ^ { N _ { \omega } } \left( \mathcal { N } _ { \delta } P _ { i } ^ { M } ( \bar { \omega } _ { h } ) - \mathrm { s g } \left[ \mathcal { N } _ { \delta } P _ { i } ^ { F } ( \bar { \omega } _ { h } ) \right] \right) ^ { 2 } .\tag{19}
$$

The sum stops at $n - 1$ , since the final memory state is not used by a downstream band. The stop gradient keeps the prefix feature spectrum as a target and trains the memory state to match it. For supervised loss $\ell _ { s u p }$ , task loss $\begin{array} { r } { \dot { \mathcal { L } } _ { \mathrm { t a s k } } = \frac { 1 } { | \mathcal { D } | } \sum _ { ( x , y ) \in \mathcal { D } } \ell _ { \mathrm { s u p } } ( \hat { y } ( x ) , y ) } \end{array}$ , the full training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } . } \end{array}\tag{20}
$$

The supervised term trains the predictor. The auxiliary term encourages the memory after each deployed prefix to store the resolved spectrum of that prefix. Frequencies outside $\dot { \Omega } _ { \mathrm { g r i d } }$ are not matched directly and are treated only through the finite grid projection. In our experiments the loss is averaged over boundaries and grid frequencies, weighted by $\lambda _ { \mathrm { a l i g n } }$ , and applied only while bands are being added (Appendix 12).

## 5 Experiments

We evaluate SQMM as a compact coordinate based reconstruction model for audio and images. The comparison targets the low parameter regime used by QIREN [85], which is the closest quantum INR baseline. All experiments were run on internal CPU cluster nodes with a mixed state simulator using PennyLane [8] and PyTorch [52]. Unless stated otherwise, models were trained with Adam [31] for 600 epochs using learning rate 0.03, betas (0.9, 0.999), epsilon $1 0 ^ { - 8 }$ , and supervised MSE loss.

Datasets. The datasets cover single signal reconstruction, dataset level audio reconstruction, and image reconstruction. The classic INR setting follows QIREN [85] and uses a Bach cello excerpt together with Astronaut, Camera, and Coffee images from scikit image [72]. Dataset level audio uses GTZAN [71], UrbanSound8K [56], and VCTK [74]. To control simulation cost, each selected audio dataset uses 10 samples per class. Since only VCTK has more than 10 speakers (classes), the first 10 are chosen. CIFAR10 [35] is used for grayscale image reconstruction on 100 images. For larger images we use all 24 Kodak images [17] and 10 DIV2K images [1], in RGB.

Baselines. Baselines include classical coordinate networks and quantum INR models. We compare against ReLU, Tanh, ReLU with random Fourier features (R+RFF) [70], SIREN [67], FINER [40], WIRE [57], KAN [41], NeRF [46], QIREN [85], and QVF [77], and against the recent INRs CAFE [30], MSA-INR [24], FLAIR [33], TUNER [48], VI3NR [34], and IGA [63]. Every classical baseline has one configuration for audio and one for images. Each uses the default settings of it official implementation, with network width and depth selected inside a fixed parameter budget and the learning rate selected from $[ 3 \cdot 1 0 ^ { - 2 } , 1 0 ^ { - 3 } ]$ . Configurations and the two baselines with adjusted settings, MSA-INR and the encoder scaled CAFE (enc.), are given in Appendix 14.5.

Metrics. Metrics are reported as reconstruction error unless otherwise stated. Audio and classic INR tables report MSE with the scaling shown in each caption. Audio analysis also reports log spectral distance. CIFAR10 reports MSE, SSIM [78], and LPIPS [84]. Params denotes the number of model parameters. All reconstruction experiments are per instance INR fitting experiments.

Table 1: MSE↓ $( \times 1 0 ^ { - 3 } )$ on sound and image representation tasks, mean $\pm \operatorname { S D } . ^ { \star }$ marks our model and its subscript the number of shots. QVF has no audio configuration. Best runs of all models are in Table 15. Best in bold.
<table><tr><td></td><td>SQMM*</td><td> $\mathrm { S Q M M _ { 2 0 4 8 } ^ { \star } }$ </td><td>QIREN</td><td>QVF</td><td>IGA</td><td>FINER</td><td>SIREN</td><td>MSA-INR</td><td>CAFE (enc.)</td></tr><tr><td>Params Cello</td><td>642 0.48±0.08</td><td>642 0.76±0.06</td><td>649 6.79±0.35</td><td></td><td>673 4.87±0.01</td><td>691 3.73±0.44</td><td>673 5.02±0.08</td><td>673 7.04±1.23</td><td>703 3.60±0.60</td></tr><tr><td>Params</td><td>654</td><td>654</td><td>657</td><td>671</td><td>687</td><td>697</td><td>687</td><td>697</td><td>709</td></tr><tr><td>Astronaut</td><td>1.12±0.10</td><td>2.86±0.53</td><td>5.90±0.49</td><td>12.61±1.65</td><td>2.01±0.53</td><td>2.80±0.31</td><td>2.15±0.22</td><td>2.03±0.28</td><td>10.46±5.92</td></tr><tr><td>Camera</td><td>0.38±0.08</td><td>1.27±0.16</td><td>2.19±0.31</td><td>4.43±1.38</td><td>1.07±0.46</td><td>1.96±0.62</td><td>0.76±0.47</td><td>1.14±0.14</td><td> $3 . 3 6 \pm 1 . 4 1$ </td></tr><tr><td>Coffee</td><td>0.78±0.05</td><td>1.79±0.16</td><td>2.52±0.67</td><td>14.38±6.80</td><td>1.63±0.42</td><td>1.85±0.31</td><td>2.26±0.40</td><td> $1 . 3 7 { \pm } 0 . 1 9$ </td><td> $5 . 1 7 { \pm } 1 . 0 6$ </td></tr></table>

Configuration. SQMM is kept near the QIREN parameter scale and uses five bands with [2, 3, 4, 4, 5] reuploads and two data qubits per band. For audio, SQMM encodes the raw coordinate with 16 dyadic sinusoidal frequencies uploaded through $R _ { Y }$ rotations. For images, SQMM uses the same circuit with 8 consecutive frequencies per axis uploaded through $R _ { Y }$ and $R _ { X }$ rotations, and RGB images only widen the output layer. All bands share a two qubit memory register, initialized from the same encoding, with controlled $R _ { Y }$ reads and XY writes. Every band is read out through its Pauli Z and ZZ expectations, and the decoder is an MLP with one hidden layer of width 5 and GELU activation. Full configurations and training details are given in Appendix 14.5.

## 5.1 Base Benchmark

The classic INR benchmark follows the QIREN protocol. The cello signal is sampled at 1000 points with stride 5, normalized in amplitude, and indexed by timestamps evenly spaced in [−1, 1]. The image task uses Astronaut, Camera, and Coffee images cropped and downsampled to $3 2 \times 3 2$ . This setting tests whether memory coupling improves the small signal regime used in prior quantum INR comparisons. For this benchmark only, models trained for 300 epochs for Cello, Camera and Coffee to match QIREN’s benchmark. The column $\mathrm { S Q M M _ { 2 0 4 8 } }$ estimates every expectation from 2048 measurement shots in training and evaluation (Appendix 14.4). Figure 1 shows the Cello reconstruction.

SQMM obtains the best reconstruction error on each sound and image representation task in Table 1. With 2048 shots SQMM stays best on the audio sample, Cello. The comparison is most informative against QIREN, FINER, and QVF because they represent the closest quantum INR baseline, a strong adaptive classical spectral baseline, and a recent quantum visual field baseline.

## 5.2 Model analysis

The mechanism ablation isolates the contribution of memory access and spectral alignment. Table 2 compares SQMM on Cello with a no memory control that gives every band two further qubits that act locally and are not shared between bands, which matches the qubit count and parameter budget of SQMM. The no memory control has a higher error than SQMM. On the full classic benchmark (Table 9), adding memory with read write substantially improves reconstruction. Directly supervising the final output spectrum with a control loss ${ \mathcal { L } } _ { \mathrm { o u t } } ( \mathrm { A p } \cdot$

Table 2: Mechanism ablation on Cello, MSE $( \times 1 0 ^ { - 3 } )$ ), mean ± SD. No memory, local qubits replaces the shared memory with two local qubits in every band and shares nothing between bands. All samples are in Table 9. Best in bold.
<table><tr><td>Setting</td><td>Params MSE↓</td><td> $( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>No memory, local qubits</td><td>650</td><td>0.53±0.06</td></tr><tr><td>Memory</td><td>642</td><td>0.52±0.08</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { o u t } }$ </td><td>642</td><td>0.54±0.06</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { a l i g n } } \left( \mathrm { O u r s } \right)$ </td><td>642</td><td>0.48±0.08</td></tr></table>

pendix 14.3) rather than the memory spectrum gives a similar error, while aligning the memory spectrum gives the best result at the same parameter count.

Table 3 shows the same effect on every sample of the base benchmark. Spectral alignment improves MSE, PSNR, and log spectral distance on Cello, and MSE, PSNR, SSIM, and LPIPS on the images. The consistency across sound and images is more important than the absolute size of any single gain. Additional analysis in Appendix 14.

Table 3: SQMM without and with $\mathcal { L } _ { \mathrm { a l i g n } }$ on the base benchmark, mean ± SD. Cello reports LSD, the images SSIM and LPIPS. Best in bold.
<table><tr><td></td><td colspan="2">Cello</td><td colspan="2">Astronaut</td><td colspan="2">Camera</td><td colspan="2">Coffee</td></tr><tr><td>Metric</td><td>No loss</td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>No loss</td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>No loss</td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>No loss</td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td></tr><tr><td> $\mathrm { M S E \downarrow ( \times 1 0 ^ { - 3 } ) }$ </td><td> $0 . 5 2 { \pm } 0 . 0 8$ </td><td>0.48±0.08</td><td>1.23±0.21</td><td> ${ \bf 1 . 1 2 \pm 0 . 1 0 }$ </td><td>0.42±0.06</td><td> $\mathbf { 0 . 3 8 \pm 0 . 0 8 }$ </td><td>0.88±0.11</td><td> $\mathbf { 0 . 7 8 \pm 0 . 0 5 }$ </td></tr><tr><td>PSNR↑ (dB)</td><td> $3 2 . 8 5 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $\mathbf { 3 3 . 2 7 \pm 0 . 7 2 }$ </td><td></td><td></td><td></td><td>35.34±0.71 35.66±0.3939.83±0.6640.28±0.94</td><td> $3 6 . 6 0 { \scriptstyle \pm 0 . 5 4 }$ </td><td>37.12±0.28</td></tr><tr><td>LSD↓ (1D)</td><td> $4 . 7 0 { \pm } 0 . 2 5 $ </td><td> $\mathbf { 4 . 4 5 \pm 0 . 1 4 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S S I M } \uparrow ( \times 1 0 ^ { - 2 } ) ( 2 \mathrm { D } )$ </td><td></td><td></td><td></td><td>99.23±0.1099.32±0.07 98.21±0.26</td><td></td><td> $\mathbf { 9 8 . 4 5 \pm 0 . 3 2 }$ </td><td></td><td>98.93±0.10 99.06±0.04</td></tr><tr><td> $\mathrm { L P I P S \downarrow ( \times 1 0 ^ { - 4 } ) ( 2 D ) }$ </td><td></td><td></td><td>3.54±0.84</td><td> $\mathbf { 2 . 6 4 \pm 0 . 6 6 }$ </td><td> $3 . 1 4 \pm 1 . 2 6$ </td><td> $\pm \mathbf { 0 . 8 2 \pm 0 . 9 0 }$ </td><td> $3 . 5 5 { \pm } 0 . 9 9$ </td><td>3.40±0.92</td></tr></table>

Table 4: Audio MSE↓ $( \times 1 0 ^ { - 3 } )$ $\mathrm { m e a n } \pm \mathrm { S D } . ^ { \star }$ marks our model. All models are in Table 16. Best in bold.
<table><tr><td></td><td>SQMM*</td><td>QIREN</td><td>FINER</td><td>SIREN</td><td>NeRF</td><td>MSA-INR</td><td>IGA</td><td>CAFE (enc.)</td></tr><tr><td>Params</td><td>642</td><td>649</td><td>691</td><td>673</td><td>698</td><td>673</td><td>673</td><td>703</td></tr><tr><td>GTZAN</td><td> $\mathbf { 4 . 7 1 \pm 0 . 1 7 }$ </td><td> $4 4 . 1 9 { \pm } 1 . 8 8 $ </td><td> $5 . 1 4 \pm 1 . 0 6$ </td><td>8.65±0.61</td><td>8.89±0.44</td><td> $4 2 . 1 1 \pm 3 . 3 9$ </td><td>8.45±0.56</td><td>6.42±1.34</td></tr><tr><td>US8K</td><td> ${ \bf 6 . 1 1 \pm 0 . 2 7 }$ </td><td> $5 1 . 8 3 { \pm } 0 . 8 6 $ </td><td> $8 . 5 6 \pm 2 . 3 5$ </td><td> $1 5 . 5 5 { \pm } 1 . 4 9$ </td><td> $1 3 . 9 6 { \pm } 0 . 7 2$ </td><td> $5 2 . 2 1 { \pm } 1 . 5 5 $ </td><td> $1 5 . 4 7 { \pm } 1 . 2 2 $ </td><td>10.83±3.75</td></tr><tr><td>VCTK</td><td> $\mathbf { 1 . 0 0 \pm 0 . 0 3 }$ </td><td> $1 0 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $1 . 6 6 { \pm } 0 . 2 1$ </td><td> $2 . 9 8 { \pm } 0 . 1 5$ </td><td> $2 . 5 9 { \pm } 0 . 1 5$ </td><td> $8 . 5 1 { \pm } 0 . 9 0 $ </td><td> $2 . 6 9 { \pm } 0 . 1 3$ </td><td>2.25±0.34</td></tr></table>

## 5.3 Audio

Audio reconstruction evaluates whether the single signal improvement transfers to varied waveform structure. GTZAN provides music, UrbanSound8K (US8K) provides environmental sounds, and VCTK provides speech. Clips are converted to mono. GTZAN uses 22.05 kHz audio, UrbanSound8K is resampled to 22.05 kHz, and VCTK is resampled to 24 kHz. A stride of 5 keeps 1024 coordinates beginning at 0.5 seconds, using the same coordinate normalization as the cello task.

Table 4 shows that SQMM remains stable across the three audio domains. The strongest competing baseline is FINER, which is consistent with the importance of adaptive spectral structure in audio INRs. SQMM improves over FINER on all datasets with fewer parameters.

## 5.4 Images

For image reconstruction, the preprocessing follows Section 5.1. We evaluate grayscale CIFAR10 using 100 images. This setting tests whether the SQMM gain extends beyond the three scikit image examples to a larger image collection. To test larger images, SQMM is fitted to the Kodak and DIV2K images from $3 2 ^ { 2 ^ { \circ } }$ to 256<sup>2</sup> pixels and at full size (Appendix 14.8).

Table 5 shows that SQMM also improves reconstruction on grayscale CIFAR10, where QIREN+ adds one spectrum layer to QIREN. SQMM achieves the lowest MSE and LPIPS and the highest SSIM with the fewest parameters of the compared models. This suggests that the memory coupled representation is not limited to the scikit image setting and transfers to a larger image collection. On Kodak and DIV2K, SQMM obtains the lowest MSE of all models at every resolution, including NeRF which holds ×1.54 the parameters of SQMM (Table 6), and Figure 3 shows example reconstructions.

## 6 Discussion

SQMM suggests that memory coupling is useful not only as an architectural addition, but as a mechanism for organizing spectral computation across quantum INR bands. In prior QINR models, Fourier features are typically produced by independent quantum feature blocks or readout circuits [85, 77]. In SQMM, the shared memory provides an intermediate quantum state through which later bands can condition their coefficients on spectra expressed by earlier bands. This connects the architectural coupling of Section 4.1 to the Fourier view of data reuploading circuits (Section 4.2 and Appendix 10) [60, 9, 83].

The alignment objective further indicates that the memory should be treated as a trainable spectral interface rather than an opaque hidden state. The ablations in Section 5.2 support this interpretation.

![](images/7ba74c403ef394d175fea9a130b2cccc168c8bec759c3203aba3b5dfd5c48071.jpg)  
Figure 3: Ground truth (top) and SQMM (bottom), CIFAR10 at $3 2 \times 3 2 .$ , Kodak and DIV2K at $1 2 \bar { 8 } \times 1 2 8$

Table 5: Grayscale CIFAR10, mean ± SD. <sup>⋆</sup> marks our model. Table 21 has all models. Best in bold.
<table><tr><td></td><td>SQMM*</td><td>QIREN+</td><td>QVF</td><td>FINER</td><td>SIREN</td><td>IGA</td><td>MSA-INR</td></tr><tr><td>Params</td><td>654</td><td>801</td><td>671</td><td>697</td><td>687</td><td>687</td><td>697</td></tr><tr><td>SSIM↑  $( \times 1 0 ^ { - 2 } )$ </td><td> $\mathbf { 9 9 . 2 1 } \pm \mathbf { 0 . 0 3 }$ </td><td> $9 7 . 6 9 { \pm } 0 . 1 9$ </td><td> $9 2 . 9 0 { \pm } 0 . 2 3 $ </td><td> $9 7 . 4 3 { \pm } 0 . 3 9$ </td><td> $9 8 . 3 7 { \pm } 0 . 3 2 $ </td><td> $9 8 . 5 6 { \pm } 0 . 2 7 $ </td><td> $9 8 . 5 9 { \pm } 0 . 1 4 $ </td></tr><tr><td> $\mathrm { L P I P S \downarrow } ( \times 1 0 ^ { - 4 } )$ </td><td> ${ \bf 3 . 4 4 \pm 0 . 2 5 }$ </td><td> $3 0 . 4 1 \pm 2 . 9 8$ </td><td> $1 0 4 . 8 4 { \pm } 8 . 3 8 $ </td><td> $1 4 . 1 7 { \scriptstyle \pm 3 . 6 9 }$ </td><td> $8 . 2 2 { \pm } 1 . 8 9$ </td><td> $7 . 6 4 \pm 1 . 7 0$ </td><td> $1 0 . 5 6 { \pm } 1 . 9 3 $ </td></tr><tr><td> $\mathrm { M S E \downarrow ( \times 1 0 ^ { - 3 } ) }$ </td><td> $\mathbf { 0 . 4 6 { \pm 0 . 0 2 } }$ </td><td> $1 . 6 7 { \pm } 0 . 1 3$ </td><td> $5 . 1 2 { \pm } 0 . 1 7$ </td><td> $1 . 4 5 { \pm } 0 . 2 1 $ </td><td> $0 . 9 8 { \pm } 0 . 2 1 $ </td><td> $0 . 9 8 { \pm } 0 . 1 8$ </td><td> $0 . 9 0 { \pm } 0 . 1 1 $ </td></tr></table>

Table 6: Kodak and DIV2K, MSE↓ $( \times 1 0 ^ { - 3 } )$ , mean ± SD. Kodak without a size is the native $7 6 8 \times 5 1 2 . ^ { \star }$ marks our model. All models are in Tables 19 and 20. Best in bold.
<table><tr><td></td><td>SQMM*</td><td>NeRF</td><td>MSA-INR</td><td>SIREN</td><td>IGA</td><td>R+RFF</td><td>FLAIR</td></tr><tr><td>Params</td><td>666</td><td>1023</td><td>747</td><td>717</td><td>717</td><td>723</td><td>723</td></tr><tr><td> $\operatorname { K o d a k } 3 2 ^ { 2 }$ </td><td> ${ \bf 1 . 5 8 \pm 0 . 1 0 }$ </td><td> $3 . 8 0 { \pm } 0 . 4 2$ </td><td> $1 . 8 5 { \pm } 0 . 0 6 $ </td><td> $2 . 7 0 { \pm } 0 . 3 3$ </td><td> $3 . 1 1 \pm 0 . 3 4$ </td><td> $3 . 3 5 { \pm } 0 . 0 7$ </td><td> $3 . 4 8 { \pm } 0 . 0 8$ </td></tr><tr><td> $\operatorname { K o d a k 6 4 ^ { 2 } }$ </td><td> ${ \pm . 4 3 \pm 0 . 0 3 }$ </td><td> $8 . 7 6 { \pm } 0 . 6 8$ </td><td> $6 . 8 4 \pm 0 . 0 6$ </td><td> $6 . 8 9 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $7 . 0 6 \pm 0 . 2 5$ </td><td> $9 . 4 7 { \pm } 0 . 1 8$ </td><td> $9 . 5 7 { \pm } 0 . 4 6 $ </td></tr><tr><td> $\operatorname { K o d a k } 1 2 8 ^ { 2 }$ </td><td> $\mathbf { 1 1 . 5 4 \pm 0 . 0 5 }$ </td><td> $1 4 . 9 4 \pm 0 . 2 6$ </td><td> $1 4 . 0 2 { \pm } 0 . 0 5$ </td><td> $1 3 . 1 8 { \pm } 0 . 4 7$ </td><td> $1 3 . 6 5 { \pm } 0 . 3 6$ </td><td> $1 6 . 8 9 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $1 7 . 5 7 { \pm } 0 . 2 8 $ </td></tr><tr><td> $\operatorname { K o d a k } 2 5 6 ^ { 2 }$ </td><td> $\mathbf { 1 8 . 5 2 \pm 0 . 0 4 }$ </td><td> $2 1 . 5 7 { \pm } 0 . 2 3 $ </td><td> $2 1 . 5 4 { \pm } 0 . 0 4$ </td><td> $2 0 . 5 7 { \pm } 0 . 5 5$ </td><td> $2 0 . 9 4 \pm 0 . 5 4$ </td><td> $2 4 . 5 3 { \pm } 0 . 3 4$ </td><td> $2 5 . 3 7 { \pm } 0 . 1 4$ </td></tr><tr><td>Kodak</td><td> $\mathbf { 3 2 . 0 8 { \pm } 0 . 1 5 }$ </td><td> $3 4 . 7 9 2 0 . 4 0$ </td><td> $3 5 . 6 4 \pm 0 . 2 6$ </td><td> $3 4 . 4 4 { \pm } 0 . 6 1$ </td><td> $3 4 . 0 4 { \pm } 0 . 5 6 $ </td><td> $3 8 . 5 4 \pm 0 . 6 4$ </td><td> $3 9 . 0 4 \pm 0 . 3 9$ </td></tr><tr><td> $\mathrm { D I V 2 K 3 2 ^ { 2 } }$ </td><td> ${ \bf 3 . 1 3 \pm 0 . 0 8 }$ </td><td> $6 . 9 1 \pm 0 . 2 0$ </td><td> $3 . 5 8 { \pm } 0 . 0 9$ </td><td> $5 . 2 2 \pm 0 . 7 4$ </td><td> $5 . 3 4 \pm 0 . 5 5$ </td><td> $6 . 1 2 { \pm } 0 . 2 2$ </td><td> $7 . 0 6 { \pm } 0 . 1 3$ </td></tr><tr><td> $\mathrm { D I V } 2 \mathrm { K } 6 4 ^ { 2 }$ </td><td> $\mathbf { 1 1 . 1 3 \pm 0 . 0 6 }$ </td><td> $1 8 . 5 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $1 3 . 6 6 { \pm } 0 . 1 1$ </td><td> $1 4 . 3 7 { \pm } 0 . 7 5 $ </td><td> $1 5 . 2 2 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $1 8 . 4 2 { \pm } 0 . 4 1$ </td><td> $1 9 . 8 3 { \pm } 0 . 2 8 $ </td></tr><tr><td> $\mathrm { D I V } 2 \mathrm { K } 1 2 8 ^ { 2 }$ </td><td> $\mathbf { 2 3 . 1 6 { \pm } 0 . 3 4 }$ </td><td> $3 0 . 8 7 { \pm } 0 . 3 8 $ </td><td> $2 7 . 3 5 { \pm } 0 . 4 1$ </td><td> $2 7 . 0 7 { \pm } 1 . 0 2 $ </td><td> $2 8 . 1 1 { \pm } 0 . 9 8 $ </td><td> $3 2 . 6 3 { \pm } 0 . 7 6$ </td><td> $3 5 . 1 8 { \pm } 0 . 3 0 $ </td></tr><tr><td> $\mathrm { D I V } 2 \mathrm { K } 2 5 6 ^ { 2 }$ </td><td> $\mathbf { 3 5 . 1 5 \pm 0 . 4 3 }$ </td><td> $4 2 . 6 8 { \pm } 1 . 2 6 $ </td><td> $3 9 . 8 4 \pm 0 . 3 1$ </td><td> $3 9 . 3 4 { \pm } 0 . 7 4$ </td><td> $4 0 . 0 2 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $4 5 . 8 0 { \pm } 0 . 9 4 \ $ </td><td> $4 8 . 1 8 { \pm } 0 . 7 0 $ </td></tr><tr><td> $\mathrm { D I V } 2 \mathrm { K } 5 1 2 ^ { 2 }$ </td><td> ${ \bf 5 4 . 1 3 \pm 0 . 4 4 }$ </td><td> $6 1 . 0 8 { \pm } 0 . 5 3 $ </td><td> $5 9 . 1 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $5 8 . 4 1 { \pm } 1 . 1 4 $ </td><td> $5 8 . 4 0 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $6 5 . 1 4 { \pm } 0 . 8 5$ </td><td> $6 8 . 1 1 \pm 1 . 3 1$ </td></tr></table>

Memory access improves reconstruction, and memory alignment gives the strongest result among the compared variants in Table 9.

These results position spectral memory as a practical design principle for compact QINRs, which could extend beyond per instance reconstruction to coordinate based settings with central spectral structure, such as higher dimensional signals, time varying representations, and physical fields [46, 80, 28]. SQMM inherits the practical constraints of parameterized QML and recurrent quantum models, such as noise sensitivity, simulator cost, and hardware dependent scaling with circuit depth and memory size [43, 7, 65, 5, 12]. No experiment was run on quantum hardware, and only finite shot estimation and coherent gate perturbations are simulated. At inference the band observables need $\mathcal { O } ( d _ { F } T / \mathrm { e } ^ { 2 } )$ shots at standard error e, while full Pauli memory probing needs $\mathcal { O } ( 4 ^ { m } T ( n - 1 ) / \mathrm { e } ^ { 2 } )$ shots per training update (Appendix 13).

## 7 Conclusion

SQMM introduces a memory coupled quantum INR where sequential data reuploading bands share a coherent quantum memory. The analysis shows that input independent memory read and write operations preserve the finite Fourier structure of data reuploading circuits while allowing later bands to condition on an operator valued spectral summary of earlier bands. SQMM achieves the best MSE among the evaluated compact INR and QINR baselines on the considered reconstruction benchmarks. Questions of quantum computational or hardware advantage remain outside the scope of this work. Overall, the results support coherent spectral memory as a useful mechanism for compact quantum implicit representations, with simulation cost and hardware measurement overhead as the main limitations.

## Acknowledgments and Disclosure of Funding

This work was supported by the D.E. Koshland Jr. Family Career Development Chair in Advanced Technologies in Electrical and Computer Engineering. We declare no competing interests.

## References

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017.

[2] Ivan Anokhin, Kirill Demochkin, Taras Khakhulin, Gleb Sterkin, Victor Lempitsky, and Denis Korzhenkov. Image generators with conditionally-independent pixel synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14278–14287, 2021.

[3] Alice Barthe and Adrián Pérez-Salinas. Gradients and frequency profiles of quantum re-uploading models. Quantum, 8:1523, November 2024. doi: 10.22331/q-2024-11-14-1523. URL https://doi.org/10. 22331/q-2024-11-14-1523.

[4] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritchman. Frequency bias in neural networks for input of non-uniform density. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 685–694. PMLR, 2020.

[5] Johannes Bausch. Recurrent quantum neural networks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/hash/0ec96be397dd6d3cf2fecb4a2d627c1c-Abstract.html.

[6] Nuri Benbarka, Timon Höfer, Andreas Zell, et al. Seeing implicit neural representations as fourier series. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2041–2050, 2022.

[7] Marcello Benedetti, Erika Lloyd, Stefan Sack, and Mattia Fiorentini. Parameterized quantum circuits as machine learning models. Quantum Science and Technology, 4(4):043001, 2019.

[8] Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Shahnawaz Ahmed, Vishnu Ajith, M Sohaib Alam, Guillermo Alonso-Linaje, B AkashNarayanan, Ali Asadi, et al. Pennylane: Automatic differentiation of hybrid quantum-classical computations. arXiv preprint arXiv:1811.04968, 2018.

[9] Berta Casas and Alba Cervera-Lierta. Multidimensional fourier series with quantum circuits. Physical Review A, 107(6):062612, 2023. doi: 10.1103/PhysRevA.107.062612.

[10] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5799–5809, 2021.

[11] Samuel Yen-Chi Chen, Tzu-Chieh Wei, Chao Zhang, Haiwang Yu, and Shinjae Yoo. Quantum convolutional neural networks for high energy physics data analysis. Physical Review Research, 4(1):013231, 2022.

[12] Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L Fang. Quantum long short-term memory. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8622–8626. IEEE, 2022.

[13] Pierre-Luc Dallaire-Demers and Nathan Killoran. Quantum generative adversarial networks. Physical Review A, 98(1):012324, 2018.

[14] Emilien Dupont, Adam Golinski, Milad Alizadeh, Yee Whye Teh, and Arnaud Doucet. Coin: Compression´ with implicit neural representations. arXiv preprint arXiv:2103.03123, 2021.

[15] Edward Farhi and Hartmut Neven. Classification with quantum neural networks on near term processors. arXiv preprint arXiv:1802.06002, 2018.

[16] Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter. Multiplicative filter networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=OmtmcPkkhT.

[17] Rich Franzen. Kodak lossless true color image suite. http://r0k.us/graphics/kodak/, 1999.

[18] Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5):1189–1232, 2001. doi: 10.1214/aos/1013203451. URL https://doi.org/10.1214/ aos/1013203451.

[19] Keisuke Fujii and Kohei Nakajima. Harnessing disordered-ensemble quantum dynamics for machine learning. Physical Review Applied, 8(2):024030, 2017.

[20] Xun Gao, Eric R Anschuetz, Sheng-Tao Wang, J Ignacio Cirac, and Mikhail D Lukin. Enhancing generative models via quantum correlations. Physical Review X, 12(2):021037, 2022.

[21] Kyle Genova, Forrester Cole, Daniel Vlasic, Aaron Sarna, William T Freeman, and Thomas Funkhouser. Learning shape templates with structured implicit functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7154–7164, 2019.

[22] Francisco Javier Gil Vidal and Dirk Oliver Theis. Input redundancy for parameterized quantum circuits. Frontiers in Physics, 8:297, 2020. doi: 10.3389/fphy.2020.00297. URL https://doi.org/10.3389/ fphy.2020.00297.

[23] Takahiro Goto, Quoc Hoan Tran, and Kohei Nakajima. Universal approximation property of quantum machine learning models in quantum-enhanced feature spaces. Physical Review Letters, 127(9):090506, 2021. doi: 10.1103/PhysRevLett.127.090506.

[24] Jufeng Han, Shu Wei, Min Wu, Lina Yu, Weijun Li, Linjun Sun, Hong Qin, and Yan Pang. Implicit neural representation with multi-scale sine activation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 21567–21575, 2026. doi: 10.1609/aaai.v40i26.39305.

[25] Vojtech Havlíˇ cek, Antonio D Córcoles, Kristan Temme, Aram W Harrow, Abhinav Kandala, Jerry Mˇ Chow, and Jay M Gambetta. Supervised learning with quantum-enhanced feature spaces. Nature, 567 (7747):209–212, 2019.

[26] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. URL https://openaccess.thecvf.com/content\_cvpr\_2016/html/He\_Deep\_Residual\_ Learning\_CVPR\_2016\_paper.html.

[27] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Predicting many properties of a quantum system from very few measurements. Nature Physics, 16:1050–1057, 2020. doi: 10.1038/s41567-020-0932-7.

[28] Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, et al. A universal pinns method for solving partial differential equations with a point source. In IJCAI, pages 3839–3846, 2022.

[29] Chiyu Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner, Thomas Funkhouser, et al. Local implicit grid representations for 3d scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6001–6010, 2020.

[30] Junbo Ke, Yangyang Xu, Chao Wang, and You-Wei Wen. Content-aware frequency encoding for implicit neural representations with Fourier-Chebyshev features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3646–3655, June 2026.

[31] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[32] Emanuel Knill, Gerardo Ortiz, and Rolando D. Somma. Optimal quantum measurements of expectation values of observables. Physical Review A, 75(1):012328, 2007. doi: 10.1103/PhysRevA.75.012328.

[33] Sukhun Ko, Seokhyun Youn, Dahyeon Kye, Kyle Min, Chanho Eom, and Jihyong Oh. FLAIR: Frequencyand locality-aware implicit neural representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 4877–4887, June 2026.

[34] Chamin Hewa Koneputugodage, Yizhak Ben-Shabat, Sameera Ramasinghe, and Stephen Gould. VI^3NR: Variance informed initialization for implicit neural representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13477–13486, June 2025.

[35] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[36] Anastasios Kyrillidis, Amir Kalev, Dohyung Park, Srinadh Bhojanapalli, and Constantine Caramanis. Provable compressed sensing quantum state tomography via non-convex methods. npj Quantum Information, 4: 36, 2018. doi: 10.1038/s41534-018-0080-4.

[37] Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-supervised nets. In Proceedings ofthe Eighteenth International Conference on Artificial Intelligence and Statistics, volume 38 of Proceedings ofMachine Learning Research, pages 562–570, 2015.

[38] Guangxi Li, Xuanqiang Zhao, and Xin Wang. Quantum self-attention neural networks for text classification. arXiv preprint arXiv:2205.05625, 2022.

[39] David B. Lindell, Dave Van Veen, Jeong Joon Park, and Gordon Wetzstein. Bacon: Band-limited coordinate networks for multiscale scene representation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16252–16262, 2022.

[40] Zhen Liu, Hao Zhu, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, and Xun Cao. FINER: Flexible spectral-bias tuning in implicit NEural representation by variable-periodic activation functions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2713–2722, 2024. doi: 10.1109/CVPR52733.2024.00262.

[41] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, and Max Tegmark. KAN: Kolmogorov–arnold networks. In International Conference on Learning Representations (ICLR), 2025.

[42] Owen Lockwood and Mei Si. Reinforcement learning with quantum variational circuit. In Proceedings of the AAAI conference on artificial intelligence and interactive digital entertainment, volume 16, pages 245–251, 2020.

[43] Jarrod R McClean, Jonathan Romero, Ryan Babbush, and Alán Aspuru-Guzik. The theory of variational hybrid quantum-classical algorithms. New Journal ofPhysics, 18(2):023023, 2016.

[44] Ali Mehrabian, Parsa Mojarad Adi, Moein Heidari, and Ilker Hacihaliloglu. Implicit neural representations with fourier kolmogorov-arnold networks. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025. arXiv:2409.09323.

[45] Hela Mhiri, Leo Monbroussou, Mario Herrero-Gonzalez, Slimane Thabet, Elham Kashefi, and Jonas Landman. Constrained and vanishing expressivity of quantum fourier models. Quantum, 9:1847, September 2025. doi: 10.22331/q-2025-09-03-1847. URL https://doi.org/10.22331/q-2025-09-03-1847.

[46] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Computer Vision – ECCV 2020, volume 12346 of Lecture Notes in Computer Science, pages 405–421. Springer, 2020. doi: 10.1007/978-3-030-58452-8\_24.

[47] Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, and Keisuke Fujii. Quantum circuit learning. Physical Review A, 98(3):032309, 2018.

[48] Tiago Novello, Diana Aldana, Andre Araujo, and Luiz Velho. Tuning the frequencies: Robust training for sinusoidal neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3071–3080, June 2025.

[49] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira Kemelmacher-Shlizerman. StyleSDF: High-resolution 3d-consistent image and geometry generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13503–13513, 2022.

[50] Giambattista Parascandolo, Heikki Huttunen, and Tuomas Virtanen. Taming the waves: sine as activation function in deep neural networks. OpenReview, 2016. URL https://openreview.net/forum?id= Sks3zF9eg.

[51] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 165–174, 2019.

[52] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, 2019.

[53] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. Data re-uploading for a universal quantum classifier. Quantum, 4:226, 2020. doi: 10.22331/q-2020-02-06-226.

[54] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International Conference on Machine Learning, pages 5301–5310. PMLR, 2019.

[55] Sameera Ramasinghe and Simon Lucey. Beyond periodicity: Towards a unifying framework for activations in coordinate-mlps. In European Conference on Computer Vision, pages 142–158. Springer, 2022.

[56] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In Proceedings ofthe 22nd ACM International Conference on Multimedia, pages 1041–1044, 2014. doi: 10.1145/2647868.2655045. URL https://doi.org/10.1145/2647868.2655045.

[57] Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, and Richard G. Baraniuk. WIRE: Wavelet implicit neural representations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18507–18516, 2023. doi: 10.1109/CVPR52729.2023.01775.

[58] Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran. Evaluating analytic gradients on quantum hardware. Physical Review A, 99(3):032331, 2019.

[59] Maria Schuld, Alex Bocharov, Krysta M Svore, and Nathan Wiebe. Circuit-centric quantum classifiers. Physical Review A, 101(3):032308, 2020.

[60] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 103(3):032430, 2021.

[61] Shayan Shekarforoush, David B. Lindell, David J. Fleet, and Marcus A. Brubaker. Residual multiplicative filter networks for multiscale reconstruction. In Advances in Neural Information Processing Systems, volume 35, pages 17713–17725, 2022.

[62] Kexuan Shi, Xingyu Zhou, and Shuhang Gu. Improved implicit neural representation with fourier reparameterized training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[63] Kexuan Shi, Hai Chen, Leheng Zhang, and Shuhang Gu. Inductive gradient adjustment for spectral bias in implicit neural representations. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 54864–54891. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/shi25a.html.

[64] S Shin, YS Teo, and H Jeong. Exponential data encoding for quantum supervised learning. Physical Review A, 107(1):012422, 2023.

[65] Sukin Sim, Peter D. Johnson, and Alán Aspuru-Guzik. Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms. Advanced Quantum Technologies, 2(12):1900070, 2019.

[66] Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. Scene representation networks: Continuous 3d-structure-aware neural scene representations. Advances in Neural Information Processing Systems, 32, 2019.

[67] Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems, volume 33, pages 7462–7473, 2020.

[68] Ivan Skorokhodov, Savva Ignatyev, and Mohamed Elhoseiny. Adversarial generation of continuous images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10753–10764, 2021.

[69] Kenneth O Stanley. Compositional pattern producing networks: A novel abstraction of development. Genetic programming and evolvable machines, 8:131–162, 2007.

[70] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems, 33: 7537–7547, 2020.

[71] George Tzanetakis and Perry Cook. Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5):293–302, 2002. doi: 10.1109/TSA.2002.800560. URL https: //doi.org/10.1109/TSA.2002.800560.

[72] Stefan Van der Walt, Johannes L Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D Warner, Neil Yager, Emmanuelle Gouillart, and Tony Yu. scikit-image: image processing in python. PeerJ, 2:e453, 2014.

[73] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

[74] Christophe Veaux, Junichi Yamagishi, and Kirsten MacDonald. CSTR VCTK Corpus: English multispeaker corpus for CSTR voice cloning toolkit, version 0.92, 2019. URL https://doi.org/10.7488/ ds/2645.

[75] Noah L Wach, Manuel S Rudolph, Fred Jendrzejewski, and Sebastian Schmitt. Data re-uploading with a single qudit. arXiv preprint arXiv:2302.13932, 2023.

[76] Hanrui Wang, Yongshan Ding, Jiaqi Gu, Yujun Lin, David Z Pan, Frederic T Chong, and Song Han. Quantumnas: Noise-adaptive search for robust quantum circuits. In 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 692–708. IEEE, 2022.

[77] Shuteng Wang, Christian Theobalt, and Vladislav Golyanik. Quantum visual fields with neural amplitude encoding. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 158535–158559. Curran Associates, Inc., 2025. doi: 10. 52202/085713-5297. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ e895155ee8e30f07f4e5e3d37475aa49-Paper-Conference.pdf.

[78] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[79] Marco Wiedmann, Maniraman Periyasamy, and Daniel D. Scherer. Fourier analysis of variational quantum circuits for supervised learning. arXiv preprint arXiv:2411.03450, 2024. URL https://arxiv.org/ abs/2411.03450.

[80] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. Learning deep time-index models for time series forecasting. In International Conference on Machine Learning, pages 37217–37237. PMLR, 2023.

[81] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. Computer Graphics Forum, 41(2):641–676, 2022. doi: 10.1111/cgf.14505.

[82] Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier analysis sheds light on deep neural networks. arXiv preprint arXiv:1901.06523, 2019.

[83] Zhan Yu, Hongshun Yao, Mujin Li, and Xin Wang. Power and limitations of single-qubit native quantum neural networks. Advances in Neural Information Processing Systems, 35:27810–27823, 2022.

[84] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[85] Jiaming Zhao, Wenbo Qiao, Peng Zhang, and Hui Gao. Quantum implicit neural representations. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 60940–60956. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/zhao24l.html.

[86] Ellen D Zhong, Tristan Bepler, Joseph H Davis, and Bonnie Berger. Reconstructing continuous distributions of 3d protein structure from cryo-em images. arXiv preprint arXiv:1909.05215, 2019.

## 8 Notation

Table 7: Notation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $x \in \mathbb { R } ^ { d } , y , \hat { y } ( x )$   $\mathrm { P E } ( { \boldsymbol { x } } ) , \gamma _ { i } ( { \boldsymbol { x } } ) , \dot { \gamma } _ { M } ( { \boldsymbol { x } } )$ </td><td>coordinate, target value, prediction positional encoding, upload variable of band i, one time memory upload</td></tr><tr><td> $n , i , L _ { i }$   $D _ { i } , q _ { i } , M , m$ </td><td>number of bands, band index, number of uploads in band i local data register of band i and its qubit count, memory register and its qubit count</td></tr><tr><td> $\eta _ { D _ { i } } , \sigma _ { M } ^ { \left( i \right) } ( x )$   $\mathcal { B } _ { i } , \mathcal { S } _ { i , \ell } , \mathcal { V } _ { i , \ell }$ </td><td>new data state of band  $i ,$  memory state after band  $i ( \sigma _ { M } ^ { ( 0 ) }$  is the initial memory) band channel, input dependent upload layer, input independent (trainable, read,</td></tr><tr><td> $\xi _ { i } ( x )$   $O _ { i , a } , f _ { i } ( x ) , d _ { f , i } , d _ { F }$ </td><td>write) block joint state of  $D _ { i } \otimes M$  after band i band observables, band features, their number per band and in total</td></tr><tr><td> $G _ { \theta }$   $\Omega _ { i , x } ^ { \mathrm { l o c } } , \Omega _ { M , x } ^ { ( i ) } , \Omega _ { f _ { i } , x }$ </td><td>classical decoder frequency support of band i, of the memory after band of the features of band i</td></tr><tr><td> $\tau _ { \mu } ^ { \left( i \right) }$ </td><td> $i ,$  matrix valued Fourier coefficient of t1  $\sigma _ { M } ^ { ( i ) }$  at frequency  $\mu$ </td></tr><tr><td> $B _ { i , \nu } , \ell _ { i , a , \nu }$ </td><td>Fourier coefficient of the band channel, linear readout functional at frequency ν</td></tr><tr><td> $c _ { \omega , a } ^ { ( i ) }$ </td><td>Fourier coefficient of feature a of band i</td></tr><tr><td> $\Omega _ { \mathrm { g r i d } } = \{ \bar { \omega } _ { h } \} _ { h = 1 } ^ { N _ { \omega } }$ </td><td>finite frequency grid of the alignment loss</td></tr><tr><td> $\widehat { a } ( u ) , \Phi , \varepsilon$ </td><td></td></tr><tr><td></td><td>estimated Fourier coefficients of a sampled signal u, design matrix, ridge</td></tr><tr><td></td><td>in all experiments)</td></tr><tr><td> $\mathcal { Q } _ { m } , Q _ { s } , \chi _ { i , s } ^ { M }$ </td><td> $\mathrm { T r } [ Q _ { s } \sigma _ { M } ^ { ( i ) } ]$  m qubit Pauli strings, one string, memory Pauli signal</td></tr><tr><td> $P _ { i } ^ { F } , P _ { i } ^ { M }$ </td><td></td></tr><tr><td></td><td>accumulated prefix feature spectrum, memory occupancy spectrum at boundary ¿</td></tr><tr><td> $\mathcal { N } _ { \delta } , \delta$ </td><td>normalization to unit mass with smoothing δ</td></tr><tr><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { a l i g n } } , \lambda _ { \mathrm { o u t } }$ </td><td>weights of and of the output spectral control</td></tr><tr><td>sg</td><td> $\mathcal { L } _ { \mathrm { a l i g n } }$  stop gradient</td></tr></table>

## 9 Quantum and Fourier Preliminaries

This appendix provides additional background to Section 3 and the definitions and proofs used by Section 4. The basic quantum notation is included to make the method readable to non QML reviewers.

Qubits, registers, and density matrices. A qubit is a two level quantum system with computational basis states |0⟩ and |1⟩. A pure one qubit state has the form $| \psi \rangle = { \dot { \alpha } } | 0 \rangle + { \dot { \beta } } | 1 \rangle$ , with $| \alpha | ^ { 2 } + | \beta | ^ { 2 } = 1$ An $n _ { q }$ qubit register lives in $( \mathbb { C } ^ { 2 } ) ^ { \otimes n _ { q } ^ { - } }$ , whose dimension is $2 ^ { n _ { q } }$ . When a subsystem may be entangled with another subsystem, it is described by a density matrix ρ, a positive semidefinite operator with $\operatorname { T r } [ \rho ] = 1$ . A register is a group of qubits. If $\rho _ { D M }$ is a joint state on a data register D with $q$ qubits and memory register M with m qubits, then $\rho _ { D M } \in \mathbb { C } ^ { 2 ^ { { q + m } } \times 2 ^ { { q + m } } }$ , and the reduced memory state is $\sigma _ { M } = \mathrm { T r } _ { D } \mathbf { \bar { [ } } \rho _ { D M } \mathbf { \bar { ] } } \in \mathbb { C } ^ { 2 ^ { m } \times 2 ^ { m } }$

Gates and channels. Quantum gates are unitary transformations. Typical single qubit rotation gates are

$$
R _ { X } ( \theta ) = e ^ { - \mathrm { i } \theta X / 2 } , \qquad R _ { Y } ( \theta ) = e ^ { - \mathrm { i } \theta Y / 2 } , \qquad R _ { Z } ( \theta ) = e ^ { - \mathrm { i } \theta Z / 2 } ,\tag{21}
$$

where $X , Y , Z \in \mathbb { C } ^ { 2 \times 2 }$ are Pauli matrices. Two qubit gates, such as CNOT, CZ, or trainable Ising couplings, create correlations between qubits. For the proofs, it is convenient to treat a circuit block as a linear channel C acting on density matrices. Unitary gates are the special case $\begin{array} { r } { \mathcal { C } ( \rho ) = U \rho U ^ { \dagger } } \end{array}$

Partial trace identities. For registers D and M, the partial trace $\mathrm { T r } _ { D }$ is the unique linear map satisfying

$$
\mathrm { T r } [ A _ { M } \ \mathrm { T r } _ { D } ( X _ { D M } ) ] = \mathrm { T r } [ ( I _ { D } \otimes A _ { M } ) X _ { D M } ]
$$

for every operator $A _ { M }$ on M. If K acts only on D, then

$$
\mathrm { T r } _ { { D } } [ ( K \otimes I _ { M } ) X _ { { D M } } ] = \mathrm { T r } _ { { D } } [ X _ { { D M } } ( K \otimes I _ { M } ) ] .
$$

This is the cyclicity of trace inside the subsystem being traced. It is used in Proposition 5.

Measurement and expectation values. For a Hermitian observable O and state $\rho ,$ the expectation value is $\langle { \cal O } \rangle = \mathrm { T r } [ { \cal O } \rho ]$ . SQMM uses expectation valued Pauli readouts as features. These features can be estimated either exactly in statevector simulation or by repeated circuit executions on hardware. They should not be interpreted as single shot selective measurement outcomes from the same coherent memory trajectory. The formal nonselective measurement statement is given in Proposition 5.

Kraus form of local measurements. A nonselective measurement on a data register D is represented by Kraus operators $\{ K _ { \zeta } \} _ { \zeta }$ on D that satisfy

$$
\sum _ { \zeta } K _ { \zeta } ^ { \dagger } K _ { \zeta } = I _ { D } .
$$

The post measurement joint state averaged over outcomes is

$$
\sum _ { \zeta } ( K _ { \zeta } \otimes I _ { M } ) \rho _ { D M } ( K _ { \zeta } ^ { \dagger } \otimes I _ { M } ) .
$$

When one conditions on a particular outcome, the corresponding branch must be divided by its probability. The distinction between averaged and conditioned states is the reason SQMM treats readouts as expectation valued probes rather than as reused single shot outcomes.

Operator spaces and channel coefficients. Let $\mathcal { L } ( \mathcal { H } )$ denote the complex vector space of linear operators on a finite dimensional Hilbert space H. A quantum channel is a completely positive trace preserving linear map on $\mathcal { L } ( \mathcal { H } )$ . Although a channel maps density matrices to density matrices, its linear extension acts on arbitrary operators in $\mathcal { L } ( \mathcal { H } )$ . This convention is used throughout the proofs when Fourier coefficient maps such as $B _ { i , \nu }$ act on matrices $\eta _ { D _ { i } } \otimes \tau _ { \mu } ^ { ( i ) }$ . The coefficient maps in a Fourier expansion of a channel need not themselves be physical channels. They are linear superoperators whose phase weighted sum is the physical channel at each coordinate value.

Data reuploading and coordinate Fourier support. Data reuploading circuits repeatedly inject the uploaded variable $\gamma _ { i } ( \boldsymbol { x } ) \in \mathbb { R } ^ { g _ { i } }$ into parameterized gates, interleaved with trainable input independent layers. For formulas in the upload domain, write $z \in \mathbb { R } ^ { g _ { i } }$ for a generic uploaded variable and reserve $\gamma _ { i } ( x )$ for the coordinate dependent value. Each input dependent upload gate is generated by a finite dimensional Hermitian operator. A typical component gate has the form

$$
U _ { i , \ell , r } ( z ) = \exp \left( - \mathrm { i } z _ { r } H _ { i , \ell , r } \right) , \qquad r = 1 , \ldots , g _ { i } ,\tag{22}
$$

where $H _ { i , \ell , r }$ is the encoding generator for upload component r. Since $H _ { i , \ell , r }$ has a finite spectrum, the matrix entries of this gate are finite sums of phases whose frequencies are the negated eigenvalues of $H _ { i , \ell , r } ,$ , so the induced channel and its expectation values have frequencies given by eigenvalue gaps. Across layers, accessible upload frequencies are finite sums of such gaps [60, 9, 83].

Linear algebra notation. For a complex matrix $A , A ^ { \dagger }$ denotes the conjugate transpose. The Frobenius norm is $\| A \| _ { F } ^ { 2 } = \operatorname { T r } ( A ^ { \dagger } A )$ . The Hilbert Schmidt inner product is

$$
\langle A , B \rangle _ { \mathrm { H S } } = \mathrm { T r } ( A ^ { \dagger } B ) .
$$

The ridge estimate in Eq. (51) is computed on the chosen finite frequency grid only. When $\varepsilon = 0 ,$ exact coefficient recovery on that grid requires Φ to have full column rank. When $\varepsilon > 0 .$ , the estimator is a stabilized projection and its coefficients are biased toward zero.

Fourier supports and conjugation. A finite operator valued Fourier expansion over coordinates has the form

$$
F ( x ) = \sum _ { \omega \in \Omega } e ^ { \mathrm { i } \left. \omega , x \right. } F _ { \omega } ,
$$

where $\Omega \subset \mathbb { R } ^ { d }$ is finite and each $F _ { \omega }$ is independent of x. The set Ω may be a superset of the nonzero coefficient support. Consequently, all support statements in the proofs are upper bounds. If $F ( x )$ has support Ω, then $F ( x ) ^ { \dagger }$ has support contained $\mathrm { i n } - \Omega ,$ , with coefficient $F _ { \omega } ^ { \dagger }$ at frequency −ω. Thus products such as $U ( x ) \dot { \eta } \dot { U } ( x ) ^ { \dagger }$ remain finite Fourier whenever $U ( x )$ is finite Fourier.

In the uploaded variable domain one may write

$$
S _ { i , \ell } ( z ) = \sum _ { \kappa \in \Omega _ { i , \ell , \gamma } } e ^ { \mathrm { i } \left. \kappa , z \right. } S _ { i , \ell , \kappa } , \qquad \Omega _ { i , \ell , \gamma } \subset \mathbb { R } ^ { g _ { i } } .\tag{23}
$$

After substituting $z = \gamma _ { i } ( x ) = A _ { i } \mathrm { P E } ( x ) + b _ { i }$ , the layer is analyzed on the coordinate domain through

$$
S _ { i , \ell } ( \gamma _ { i } ( x ) ) = \sum _ { \nu \in \Omega _ { i , \ell , x } } e ^ { \mathrm { i } \langle \nu , x \rangle } S _ { i , \ell , \nu } .\tag{24}
$$

For affine raw coordinate uploads this expansion is exact, with coordinate frequencies induced by $A _ { i } ^ { \top } \kappa .$ . For fixed sinusoidal positional encodings, the analysis can instead be viewed in the lifted coordinate $\operatorname { P E } ( x )$ , or projected onto a finite coordinate grid. If an implementation places a trigonometric polynomial inside a rotation angle, the resulting coordinate expansion may be countable rather than finite. In that case the same support algebra applies to any finite truncation, and the alignment loss uses exactly such a finite working grid.

Pauli basis normalization. The m qubit Pauli strings form an orthogonal basis of $\mathcal { L } ( ( \mathbb { C } ^ { 2 } ) ^ { \otimes m } )$ under the Hilbert Schmidt inner product. With the unnormalized convention used here,

$$
\mathrm { T r } ( Q _ { s } Q _ { s ^ { \prime } } ) = 2 ^ { m } { \bf 1 } \{ s = s ^ { \prime } \} .
$$

Hence every operator $A$ on m qubits admits the expansion

$$
A = 2 ^ { - m } \sum _ { Q _ { s } \in \mathcal { Q } _ { m } } \mathrm { T r } ( Q _ { s } A ) Q _ { s } .
$$

Parseval identity for this orthogonal basis gives

$$
\| A \| _ { F } ^ { 2 } = 2 ^ { - m } \sum _ { Q _ { s } \in \mathcal { Q } _ { m } } | \operatorname { T r } ( Q _ { s } A ) | ^ { 2 } .
$$

## 10 Proofs for Spectral Memory Propagation

This section proves the statements summarized in Section 4.2. The proofs use the coordinate domain notation of the method. Thus $x \in \mathbb { R } ^ { d } , \gamma _ { i } ( x ) \in \mathbb { R } ^ { g } \mathrm { { : } }$ <sup>i</sup>, coordinate frequencies lie in $\mathbb { R } ^ { d }$ , and phases have the form $e ^ { \mathrm { i } \langle \omega , x \rangle }$ . All local supports $\Omega _ { i , x } ^ { \mathrm { l o c } }$ are effective coordinate supports after composing the uploaded variable map $x \mapsto \gamma _ { i } ( x )$ . Unbarred frequencies $\mu , \nu , \omega$ denote theoretical coordinate frequencies. Barred frequencies $\bar { \omega } _ { h }$ are reserved for the finite grid used in Appendix 12. The symbol $\tau _ { \omega } ^ { \left( i \right) }$ always denotes the matrix Fourier coefficient of the memory density matrix at frequency $\omega .$ Throughout, $D _ { i }$ denotes the data register of band i, M the shared memory register, $I _ { M }$ the identity on memory, $\eta _ { D _ { \mathrm { i } } }$ the fixed new data state, $S _ { i , \ell } ( \gamma _ { i } ( x ) ) ,$ ) the input dependent encoding layers, and $\nu _ { i , \ell }$ the input independent trainable layers.

Lemma 2 (closure under composition). Let $\mathcal { A } ( x )$ and $\mathcal C ( x )$ be channel families with finite coordinate domain Fourier expansions

$$
\mathcal { A } ( x ) = \sum _ { \alpha \in \Omega _ { \cal A } } e ^ { \mathrm { i } \langle \alpha , x \rangle } \mathcal { A } _ { \alpha } , \qquad \mathcal { C } ( x ) = \sum _ { \beta \in \Omega _ { \cal C } } e ^ { \mathrm { i } \langle \beta , x \rangle } \mathcal { C } _ { \beta } ,\tag{25}
$$

where $\Omega _ { \mathcal { A } } , \Omega _ { \mathcal { C } } \subset \mathbb { R } ^ { d }$ are finite and each coefficient map is independent $o f x .$ Then

$$
( \boldsymbol { A } \circ \mathcal { C } ) ( \boldsymbol { x } ) = \sum _ { \omega \in \Omega _ { \boldsymbol { A } } + \Omega _ { \boldsymbol { C } } } e ^ { \mathrm { i } \langle \omega , \boldsymbol { x } \rangle } \left( \sum _ { \alpha + \beta = \omega } \boldsymbol { A } _ { \alpha } \circ \mathcal { C } _ { \beta } \right) .\tag{26}
$$

In particular, the composition isfinite Fourier on the coordinate domain.

Proof. Expanding the composition gives

$$
( \mathcal { A } \circ \mathcal { C } ) ( x ) = \sum _ { \alpha \in \Omega _ { \mathcal { A } } } \sum _ { \beta \in \Omega _ { \mathcal { C } } } e ^ { \mathrm { i } \left. \alpha + \beta , x \right. } \mathcal { A } _ { \alpha } \circ \mathcal { C } _ { \beta } .\tag{27}
$$

Grouping terms with the same total frequency $\omega = \alpha + \beta$ gives the stated expression.

Proposition 1 (band closure under memory read and write insertion). Consider a bare data reuploading band

$$
\begin{array} { r } { \mathcal { B } _ { i } ( \gamma _ { i } ( x ) ) = \mathcal { W } _ { i , L _ { i } + 1 } \circ S _ { i , L _ { i } } ( \gamma _ { i } ( x ) ) \circ \mathcal { W } _ { i , L _ { i } } \circ \cdot \cdot \circ S _ { i , 1 } ( \gamma _ { i } ( x ) ) \circ \mathcal { W } _ { i , 1 } , } \end{array}\tag{28}
$$

where each $S _ { i , \ell } ( \gamma _ { i } ( x ) )$ is input dependent and each $\mathcal { W } _ { i , \ell }$ is input independent. $A f t e r$ inserting any input independent memory read and memory write layers, the resulting band can still be written as

$$
\begin{array} { r } { \mathcal { B } _ { i } ( \gamma _ { i } ( x ) ) = \mathcal { V } _ { i , L _ { i } + 1 } \circ S _ { i , L _ { i } } ( \gamma _ { i } ( x ) ) \circ \mathcal { V } _ { i , L _ { i } } \circ \cdots \circ S _ { i , 1 } ( \gamma _ { i } ( x ) ) \circ \mathcal { V } _ { i , 1 } , } \end{array}\tag{29}
$$

for suitable input independent blocks $\gamma _ { i , \ell }$ . Consequently, memory coupling preserves the reuploading form on the enlarged register $D _ { i } \otimes M$

Proof. Each inserted read or write layer is independent of x and $\gamma _ { i } ( x )$ , so it can be absorbed into the nearest neighboring input independent block. For example, a read layer R placed after $S _ { i , \ell } ( \gamma _ { i } ( x ) )$ can be absorbed by redefining $\mathcal { V } _ { i , \ell + 1 } = \mathcal { W } _ { i , \ell + 1 } \circ \mathcal { R }$ . A write layer W placed after the last encoding layer is absorbed into the closing block $\mathscr { V } _ { i , L _ { i } + 1 }$ . The same argument applies to memory only layers. The only input dependent objects remain the encoding channels $S _ { i , \ell } ( \gamma _ { i } ( x ) )$ , so the alternating reuploading structure is unchanged. □

Corollary 1 (finite Fourier stack). Ifevery band $B _ { j } ( \gamma _ { j } ( x ) )$ has afinite effective coordinate Fourier expansion with support $\Omega _ { j , x } ^ { \mathrm { l o c } }$ , then every finite prefix channel through the first i bands has a finite effective coordinate Fourier expansion.

Proof. The case $i = 1$ is immediate. The inductive step follows from Lemma 2 applied to the next band and the existing prefix channel. Input independent initialization, partial trace operations, and tensoring with a fixed new data state are linear maps independent of $x ,$ so they do not introduce new coordinate frequencies. An encoded initialization has a finite expansion by Proposition 2. □

Proposition 2 (memory initialization with optional initial encoding). Let the memory register have m qubits. If the initial memory state is $\eta _ { M } = \overline { { | 0 \rangle \langle 0 | ^ { \otimes m } } }$ or $\eta _ { M } = | \bar { + } \rangle \langle + | ^ { \otimes m }$ , and no input dependent gate is applied to memory before the first band, then

$$
\sigma _ { M } ^ { ( 0 ) } ( x ) = \eta _ { M } , \qquad \Omega _ { M , x } ^ { ( 0 ) } = \{ 0 \} .\tag{30}
$$

If an input dependent memory encoding with the positional encoding of the bands is applied once at the beginning, define $\gamma _ { M } ( x ) = A _ { M } \mathrm { { P E } } ( x ) + b _ { M }$ , let $U _ { M } ( \gamma _ { M } ( x ) )$ ) be a finite product of memory upload gates, for example gates $R _ { Y } ( ( \gamma _ { M } ( x ) ) _ { k } )$ on memory qubits, and write $U _ { M } \mathbf { \bar { ( } } x ) : = \dot { U _ { M } } ( \gamma _ { M } ( \dot { x } ) )$ . Then

$$
\sigma _ { M } ^ { ( 0 ) } ( x ) = U _ { M } ( x ) \eta _ { M } U _ { M } ( x ) ^ { \dagger } = \sum _ { \mu \in \Omega _ { M , x } ^ { ( 0 ) } } e ^ { \mathrm { i } \langle \mu , x \rangle } \tau _ { \mu } ^ { ( 0 ) }\tag{31}
$$

for somefinite support $\Omega _ { M , x } ^ { ( 0 ) } \subset \mathbb { R } ^ { d }$ and matrices $\tau _ { \mu } ^ { ( 0 ) } \in \mathbb { C } ^ { 2 ^ { m } \times 2 ^ { m } }$ , or approximatelyfor a truncated support. The memory recursion and the downstream readout proof remain valid with this nontrivial initial support.

Proof. For $\eta _ { M } = | 0 \rangle \langle 0 | ^ { \otimes m }$ or $\eta _ { M } = | + \rangle \langle + | ^ { \otimes m }$ without input dependent preprocessing, $\sigma _ { M } ^ { ( 0 ) } ( x )$ is constant in x. A constant function has the Fourier expansion $e ^ { \mathrm { i } \langle 0 , x \rangle } \eta _ { M }$ , so its support is {0}.

Now suppose that the memory is passed through an input dependent unitary $U _ { M } ( \gamma _ { M } ( x ) )$ ) once at the beginning, with $\gamma _ { M } ( x ) = A _ { M } \mathrm { { P E } } ( x ) + b _ { M }$ . For a single gate $R _ { Y } ( ( \gamma _ { M } ( x ) ) _ { k } )$

$$
R _ { Y } ( ( \gamma _ { M } ( x ) ) _ { k } ) = \cos \left( \frac { ( \gamma _ { M } ( x ) ) _ { k } } { 2 } \right) I - \mathrm { i } \sin \left( \frac { ( \gamma _ { M } ( x ) ) _ { k } } { 2 } \right) Y .\tag{32}
$$

Under the same finite or truncated coordinate support assumption used for the data uploading gates, each matrix entry of $R _ { Y } ( ( \gamma _ { M } ( x ) ) _ { k } )$ has a finite or truncated Fourier expansion over $x .$ . Products of finitely many such memory encoding gates also have finite or truncated expansions by the argument of Lemma 2. The density matrix $U _ { M } ( x ) \eta _ { M } U _ { M } ( x ) ^ { \dagger }$ is formed by products of entries from ${ \bar { U } } _ { M } ( x )$ and their complex conjugates. Therefore it has a finite or truncated operator valued Fourier expansion with some support $\Omega _ { M , x } ^ { ( 0 ) }$

The proof of Proposition 3 only assumes that the incoming memory has an operator valued Fourier expansion. It does not require the base support to be {0}. Therefore the same recursion applies when the memory is initialized by |0⟩ or |+⟩ and then receives an initial positional encoding driven by the same $\operatorname { P E } ( x )$ used by the bands. The only change is that the global support starts from $\Omega _ { M , x } ^ { ( 0 ) }$ rather than {0}, so the theory is unchanged after replacing the trivial base support by this encoded base support. □

Proposition 3 (operator valued Fourier recursion on memory). Assume the memory entering band i+1 has theform

$$
\sigma _ { M } ^ { ( i ) } ( x ) = \sum _ { \mu \in \Omega _ { M , x } ^ { ( i ) } } e ^ { \mathrm { i } \langle \mu , x \rangle } \tau _ { \mu } ^ { ( i ) } , \qquad \Omega _ { M , x } ^ { ( i ) } \subset \mathbb { R } ^ { d } , \qquad \tau _ { \mu } ^ { ( i ) } \in \mathbb { C } ^ { 2 ^ { m } \times 2 ^ { m } } ,\tag{33}
$$

and that band i + 1 has the effective coordinate domain channel expansion

$$
\mathcal { B } _ { i + 1 } ( \gamma _ { i + 1 } ( x ) ) = \sum _ { \nu \in \Omega _ { i + 1 , x } ^ { \mathrm { l o c } } } e ^ { \mathrm { i } \left. \nu , x \right. } \mathcal { B } _ { i + 1 , \nu } , \qquad \Omega _ { i + 1 , x } ^ { \mathrm { l o c } } \subset \mathbb { R } ^ { d } .\tag{34}
$$

Then the outgoing memory state

$$
\sigma _ { M } ^ { ( i + 1 ) } ( x ) = \mathrm { T r } _ { D _ { i + 1 } } \Big [ \mathcal { B } _ { i + 1 } ( \gamma _ { i + 1 } ( x ) ) \big ( \eta _ { D _ { i + 1 } } \otimes \sigma _ { M } ^ { ( i ) } ( x ) \big ) \Big ]\tag{35}
$$

has the expansion

$$
\sigma _ { M } ^ { ( i + 1 ) } ( x ) = \sum _ { \omega \in \Omega _ { M , x } ^ { ( i + 1 ) } } e ^ { \mathrm { i } \left. \omega , x \right. } \tau _ { \omega } ^ { ( i + 1 ) } ,\tag{36}
$$

where

$$
\tau _ { \omega } ^ { ( i + 1 ) } = \sum _ { \mu + \nu = \omega } \mathrm { T r } _ { D _ { i + 1 } } \Big [ \mathcal { B } _ { i + 1 , \nu } \big ( \eta _ { D _ { i + 1 } } \otimes \tau _ { \mu } ^ { ( i ) } \big ) \Big ] .\tag{37}
$$

Consequently,

$$
\Omega _ { M , x } ^ { ( i + 1 ) } \subseteq \Omega _ { M , x } ^ { ( i ) } + \Omega _ { i + 1 , x } ^ { \mathrm { l o c } } .\tag{38}
$$

Proof. Substituting the two Fourier expansions into the memory update gives

$$
\sigma _ { M } ^ { ( i + 1 ) } ( x ) = \mathrm { T r } _ { D _ { i + 1 } } \Biggl [ \left( \sum _ { \nu } e ^ { \mathrm { i } \langle \nu , x \rangle } B _ { i + 1 , \nu } \right) \left( \eta _ { D _ { i + 1 } } \otimes \sum _ { \mu } e ^ { \mathrm { i } \langle \mu , x \rangle } \tau _ { \mu } ^ { ( i ) } \right) \Biggr ]\tag{39}
$$

$$
= \sum _ { \mu , \nu } e ^ { \mathrm { i } \left. \mu + \nu , x \right. } \mathrm { T r } _ { D _ { i + 1 } } \Big [ \mathcal { B } _ { i + 1 , \nu } \big ( \eta _ { D _ { i + 1 } } \otimes \tau _ { \mu } ^ { ( i ) } \big ) \Big ] .\tag{40}
$$

Regrouping terms by $\omega = \mu + \nu$ gives Eq. (37) and the support bound.

Proposition 4 (downstream readout coefficients). Under the assumptions ofProposition 3, let $O _ { i + 1 , a }$ be a Hermitian observable usedforfeature component a ofband $i + 1$ . Thefeature extraction map

$$
f _ { i + 1 , a } ( x ) = \mathrm { T r } \Bigl [ ( O _ { i + 1 , a } \otimes I _ { M } ) { \cal B } _ { i + 1 } ( \gamma _ { i + 1 } ( x ) ) \bigl ( \eta _ { D _ { i + 1 } } \otimes \sigma _ { M } ^ { ( i ) } ( x ) \bigr ) \Bigr ]\tag{41}
$$

admits a finite or truncated Fourier expansion whose support is contained in $\Omega _ { M , x } ^ { ( i ) } + \Omega _ { i + 1 , x } ^ { \mathrm { l o c } } .$ . Namely,

$$
f _ { i + 1 , a } ( x ) = \sum _ { \omega \in \Omega _ { M , x } ^ { ( i ) } + \Omega _ { i + 1 , x } ^ { \mathrm { l o c } } } c _ { \omega , a } ^ { ( i + 1 ) } e ^ { \mathrm { i } \left. \omega , x \right. } \mathrm { , }\tag{42}
$$

with coefficients

$$
c _ { \omega , a } ^ { ( i + 1 ) } = \sum _ { \mu + \nu = \omega } \ell _ { i + 1 , a , \nu } ( \tau _ { \mu } ^ { ( i ) } ) , \qquad \ell _ { i + 1 , a , \nu } ( X ) = \mathrm { T r } \big [ ( O _ { i + 1 , a } \otimes I _ { M } ) \mathcal { B } _ { i + 1 , \nu } \bigl ( \eta _ { D _ { i + 1 } } \otimes X \bigr ) \big ] .\tag{43}
$$

Here $\ell _ { i + 1 , a , \nu }$ is the readout functional associated with observable $O _ { i + 1 , a }$ and band frequency coefficient $B _ { i + 1 , \nu }$ . It takes a memory side matrix $X ,$ appends the fixed data state, applies the channel coefficient, and returns a scalar expectation value.

Proof. The proof repeats the expansion in Proposition 3, but applies the feature extraction functional rather than tracing out the data register. The functional is linear because tensoring with a fixed operator, applying $B _ { i + 1 , \nu }$ , and taking a trace are linear in X. Linearity gives Eq. (43). □

Corollary 2 (global support bound). Ifevery band satisfies the assumptions ofProposition 3 and the memory is initialized independently ofx, so $\Omega _ { M , x } ^ { ( 0 ) } = \{ 0 \}$ , then

$$
\Omega _ { M , x } ^ { ( i ) } \subseteq \sum _ { j = 1 } ^ { i } \Omega _ { j , x } ^ { \mathrm { l o c } } , \qquad \Omega _ { f _ { i + 1 } , x } \subseteq \sum _ { j = 1 } ^ { i + 1 } \Omega _ { j , x } ^ { \mathrm { l o c } } .\tag{44}
$$

If the memory receives an initial input dependent encoding as in Proposition 2, then

$$
\Omega _ { M , x } ^ { ( i ) } \subseteq \Omega _ { M , x } ^ { ( 0 ) } + \sum _ { j = 1 } ^ { i } \Omega _ { j , x } ^ { \mathrm { l o c } } , \qquad \Omega _ { f _ { i + 1 } , x } \subseteq \Omega _ { M , x } ^ { ( 0 ) } + \sum _ { j = 1 } ^ { i + 1 } \Omega _ { j , x } ^ { \mathrm { l o c } } .\tag{45}
$$

Proof. Apply Proposition 3 recursively. For input independent initialization, Proposition 2 gives $\Omega _ { M , x } ^ { ( 0 ) } = \{ 0 \}$ . For an initial memory encoding driven by the same $\operatorname { P E } ( x )$ used by the bands, Proposition 2 gives the finite or truncated base support $\Omega _ { M , x } ^ { ( 0 ) }$ . Applying Proposition 4 to the next readout gives the output support bound. □

## 11 Readout Probes and Memory Semantics

The main text separates coherent memory propagation from expectation valued readout. This section states the formal measurement fact behind that separation.

Proposition 5 (memory state after nonselective local readout). Let $\rho _ { D M } \in \mathbb { C } ^ { 2 ^ { q + m } \times 2 ^ { q + m } }$ be the joint post write state of a data register D with q qubits and memory register M with m qubits, and let $\sigma _ { M } = \mathrm { T r } _ { D } [ \bar { \rho _ { D M } } ]$ . Let a nonselective measurement act only on D, with Kraus operators $\{ K _ { \zeta } \} _ { \zeta }$ satisfying $\begin{array} { r } { \sum _ { \zeta } K _ { \zeta } ^ { \dagger } K _ { \zeta } = I _ { D } } \end{array}$ . Then the reduced memory state after averaging over outcomes is unchanged.

$$
\sum _ { \zeta } \mathrm { T r } _ { D } \Big [ \big ( K _ { \zeta } \otimes I _ { M } \big ) \rho _ { D M } \big ( K _ { \zeta } ^ { \dagger } \otimes I _ { M } \big ) \Big ] = \sigma _ { M } .\tag{46}
$$

Proof. Using cyclicity inside the traced subsystem,

$$
\begin{array} { l } { { \displaystyle \sum _ { \zeta } \mathrm { T r } _ { D } \Big [ \big ( K _ { \zeta } \otimes I _ { M } \big ) \rho _ { D M } \big ( K _ { \zeta } ^ { \dagger } \otimes I _ { M } \big ) \Big ] } \ ~ } \\ { { \displaystyle \qquad = \sum _ { \zeta } \mathrm { T r } _ { D } \Big [ \rho _ { D M } \big ( K _ { \zeta } ^ { \dagger } K _ { \zeta } \otimes I _ { M } \big ) \Big ] = \mathrm { T r } _ { D } \Bigg [ \rho _ { D M } \left( \sum _ { \zeta } K _ { \zeta } ^ { \dagger } K _ { \zeta } \otimes I _ { M } \right) \Bigg ] = \mathrm { T r } _ { D } \big [ \rho _ { D M } \Big ] . } } \end{array}\tag{47}
$$

(48)

Selective outcomes. If one conditions on a particular outcome $\zeta ,$ the memory state becomes

$$
\sigma _ { M | \zeta } = \frac { \mathrm { T r } _ { D } [ ( K _ { \zeta } \otimes I _ { M } ) \rho _ { D M } ( K _ { \zeta } ^ { \dagger } \otimes I _ { M } ) ] } { p _ { \zeta } } , \qquad p _ { \zeta } = \mathrm { T r } [ ( K _ { \zeta } ^ { \dagger } K _ { \zeta } \otimes I _ { M } ) \rho _ { D M } ] ,\tag{49}
$$

which generally depends on ζ. Therefore, SQMM features are expectation values estimated from readout probes or separate repeated executions, not single shot selective outcomes whose post measurement memory state is reused downstream.

## 12 Details of Spectral Memory Alignment

This section expands the vector notation used in Section 4.3. The propagation lemma uses unbarred theoretical frequencies, while the loss is evaluated only on the finite grid $\Omega _ { \mathrm { g r i d } } = \{ \bar { \omega } _ { 1 } , \dots , \bar { \omega } _ { N _ { \omega } } \} \subset$ R<sup>d</sup>.

Let $x _ { 1 } , \dots , x _ { T } \in  { \mathbb { R } } ^ { d }$ be the coordinate grid used only for the auxiliary spectral loss. For a sampled scalar signal $\boldsymbol { u } = ( u ( \boldsymbol { x } _ { 1 } ) , \ldots , u ( \boldsymbol { x } _ { T } ) ) ^ { \top } \in \mathbb { C } ^ { T }$ , define the Fourier design matrix $\Phi \in \mathbb { C } ^ { T \times N _ { \omega } }$ by

$$
\Phi _ { t h } = e ^ { \mathrm { i } \left. \bar { \omega } _ { h } , x _ { t } \right. } .\tag{50}
$$

Coefficients on $\Omega _ { \mathrm { g r i d } }$ are estimated by ridge stabilized least squares,

$$
\widehat { a } ( u ) = ( \Phi ^ { \dagger } \Phi + \varepsilon I _ { N _ { \omega } } ) ^ { - 1 } \Phi ^ { \dagger } u .\tag{51}
$$

Here $I _ { N _ { \omega } } \in \mathbb { R } ^ { N _ { \omega } \times N _ { \omega } }$ is the identity matrix. In exact simulation, ε can be set to zero when Φ has full column rank. For finite shot estimates or ill conditioned grids, $\varepsilon > 0$ stabilizes the coefficient estimate.

For feature component a of band j, define the sampled signal

$$
\boldsymbol { u } _ { j , a } ^ { F } = ( f _ { j , a } ( \boldsymbol { x } _ { 1 } ) , \ldots , f _ { j , a } ( \boldsymbol { x } _ { T } ) ) ^ { \top } , \qquad \widehat { c } _ { j , a , h } ^ { F } = \left[ \widehat { a } ( \boldsymbol { u } _ { j , a } ^ { F } ) \right] _ { h } .\tag{52}
$$

Collecting all feature components of band $j$ at grid frequency $\bar { \omega } _ { h }$ gives

$$
\widehat { \pmb { c } } _ { j , h } ^ { F } = ( \widehat { \pmb { c } } _ { j , 1 , h } ^ { F } , \dots , \widehat { \pmb { c } } _ { j , d _ { f , j } , h } ^ { F } ) \in \mathbb { C } ^ { d _ { f , j } } .\tag{53}
$$

The prefix feature occupancy used in the main text is therefore

$$
P _ { i } ^ { F } ( \bar { \omega } _ { h } ) = \sum _ { j = 1 } ^ { i } \| \widehat { \pmb { c } } _ { j , h } ^ { F } \| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { i } \sum _ { a = 1 } ^ { d _ { f , j } } | \widehat { c } _ { j , a , h } ^ { F } | ^ { 2 } .\tag{54}
$$

This is an occupancy measure over extracted quantum features, not a spectrum of a linear layer or deployed decoder contribution. The occupancy target records which frequencies have been represented by earlier band features before the final decoder.

To estimate the memory spectrum, use the unnormalized Pauli string basis

$$
{ \mathcal { Q } } _ { m } = \{ I , X , Y , Z \} ^ { \otimes m } , \qquad | { \mathcal { Q } } _ { m } | = 4 ^ { m } , \qquad \operatorname { T r } ( Q _ { s } Q _ { s ^ { \prime } } ) = 2 ^ { m } { \bf 1 } \{ s = s ^ { \prime } \} .\tag{55}
$$

For each basis element, define the sampled memory signal

$$
\boldsymbol { u } _ { i , s } ^ { M } = ( \mathrm { T r } [ Q _ { s } \sigma _ { M } ^ { ( i ) } ( \boldsymbol { x } _ { 1 } ) ] , \dots , \mathrm { T r } [ Q _ { s } \sigma _ { M } ^ { ( i ) } ( \boldsymbol { x } _ { T } ) ] ) ^ { \top } , \qquad \widehat { c } _ { i , s , h } ^ { M } = \left[ \widehat { a } ( \boldsymbol { u } _ { i , s } ^ { M } ) \right] _ { h } .\tag{56}
$$

Collecting the Pauli probe coefficients at grid frequency $\hat { \omega } _ { h }$ gives

$$
\widehat { \pmb { c } } _ { i , h } ^ { M } = ( \widehat { \pmb { c } } _ { i , s , h } ^ { M } ) _ { { Q } _ { s } \in { \mathscr { Q } } _ { m } } \in \mathbb { C } ^ { 4 ^ { m } } .\tag{57}
$$

The memory occupancy spectrum is

$$
P _ { i } ^ { M } ( \bar { \omega } _ { h } ) = 2 ^ { - m } \sum _ { Q _ { s } \in \mathcal { Q } _ { m } } | \widehat { c } _ { i , s , h } ^ { M } | ^ { 2 } .\tag{58}
$$

If the memory state has the exact expansion

$$
\sigma _ { M } ^ { ( i ) } ( x ) = \sum _ { \omega \in \Omega _ { M , x } ^ { ( i ) } } e ^ { \mathrm { i } \left. \omega , x \right. } \tau _ { \omega } ^ { ( i ) } ,\tag{59}
$$

then, when every frequency of $\Omega _ { M , x } ^ { ( i ) }$ is a discrete Fourier frequency of the sampling grid and $\varepsilon = 0$

$$
\widehat c _ { i , s , h } ^ { M } = \mathrm { T r } [ Q _ { s } \tau _ { \bar { \omega } _ { h } } ^ { ( i ) } ] .\tag{60}
$$

The Pauli strings form an orthogonal operator basis under the Hilbert Schmidt inner product. For any operator A on m qubits,

$$
A = 2 ^ { - m } \sum _ { Q _ { s } \in \mathcal { Q } _ { m } } \mathrm { T r } ( Q _ { s } A ) Q _ { s } ,\tag{61}
$$

and therefore

$$
\| A \| _ { F } ^ { 2 } = \operatorname { T r } ( A ^ { \dagger } A ) = 2 ^ { - m } \sum _ { Q _ { s } \in { \mathcal { Q } } _ { m } } | \operatorname { T r } ( Q _ { s } A ) | ^ { 2 } .\tag{62}
$$

Applying this identity to $A = \tau _ { \bar { \omega } _ { h } } ^ { ( i ) }$ gives

$$
P _ { i } ^ { M } ( \bar { \omega } _ { h } ) = \| \tau _ { \bar { \omega } _ { h } } ^ { ( i ) } \| _ { F } ^ { 2 }\tag{63}
$$

up to finite grid estimation error. The identity Pauli string contributes only to the zero frequency trace component in the ideal normalized state setting.

Approximate and truncated spectra. The exact identity above requires an exact finite expansion on the coordinate domain and a frequency grid containing the frequency of interest. If the composed circuit has countable or continuous spectral content, or if relevant frequencies are absent from $\Omega _ { \mathrm { g r i d } }$ the estimator computes the ridge projection of the sampled memory signals onto the chosen grid. In that case $P _ { i } ^ { M }$ is a proxy for the resolved memory spectrum rather than the full memory spectrum. The resulting limitations are standard finite grid spectral limitations, such as, off grid frequencies can leak into nearby grid coefficients, unresolved high frequencies are ignored, insufficient or nonuniform sampling can cause aliasing, and the ridge parameter can bias small coefficients. The propagation lemma still describes the algebra of the exact or truncated expansion being analyzed, but the alignment loss only constrains the portion visible through the selected probes, samples, and grid.

On hardware, estimating the full Pauli basis is expensive. Practical implementations may use a fixed subset of probes or randomized Pauli measurements, which gives a lower cost proxy for the same memory spectrum.

For any nonnegative spectrum $R : \Omega _ { \mathrm { g r i d } }  \mathbb { R } _ { \geq 0 }$ , the normalized spectrum used by the alignment loss is

$$
\mathcal { N } _ { \delta } R ( \bar { \omega } _ { h } ) = \frac { R ( \bar { \omega } _ { h } ) + \delta } { \sum _ { g = 1 } ^ { N _ { \omega } } R ( \bar { \omega } _ { g } ) + N _ { \omega } \delta } .\tag{64}
$$

The stop gradient in Eq. (19) is applied only to the normalized prefix feature occupancy, not to the memory occupancy.

## 13 Limitations

Spectral approximation and memory alignment. The spectral propagation result is exact when the composed memory state admits the finite coordinate Fourier expansion in Eq. (8). In implementations with sinusoidal positional encodings, nonlinear angle maps, finite sampling, or hardware estimates, this expansion is generally analyzed through a finite truncation or grid projection. The alignment objective in Section 4.3 inherits this approximation, since it matches the memory spectrum and prefix feature spectrum only on the chosen grid $\Omega _ { \mathrm { g r i d } }$ and only through the selected Pauli memory probes. Appendix 12 makes this proxy interpretation explicit. Frequencies outside the grid are not directly constrained. They can leak into nearby coefficients, alias under the sampled coordinate set, remain unresolved when they are too high frequency for the grid, or be suppressed by the ridge term in Eq. (51). Therefore $\dot { \mathcal { L } } _ { \mathrm { a l i g n } }$ should be interpreted as a practical mechanism for shaping the resolved memory spectrum.

Hardware execution and measurement cost. SQMM also inherits practical limitations common to parameterized QML and recurrent quantum models. Pauli readout features are expectation values, so hardware execution requires repeated circuit runs with fixed parameters. For a band feature $f _ { i , a } ( x )$ the circuit is executed up to band $i , O _ { i , a }$ is measured on $D _ { i } ,$ and outcomes are averaged across shots. The measured state is not reused for later bands. Downstream features use separate executions that preserve the coherent memory path without conditioning on single shot outcomes, as formalized in Appendix 11. This creates a measurement overhead. Estimating the $d _ { F }$ readout observables at one coordinate uses $\mathcal { O } ( d _ { F } / \mathrm { e } ^ { 2 } )$ shots for standard error $\mathrm { e } ,$ so a pass over $T$ coordinates uses $\mathcal { O } ( d _ { F } T / \mathrm { e } ^ { 2 } )$ shots [32]. Full spectral alignment is more expensive and uses $\mathcal { O } ( 4 ^ { m } T ( n - 1 ) / \mathrm { e } ^ { 2 } )$ shots per training update for an m qubit memory, $T$ sampled coordinates and $n - 1$ boundaries [36], unless subset probes or shadow based estimators are used [27]. In addition, noise affects both the memory evolution and the readout estimates. Appendix 14.2 evaluates controlled circuit perturbations and observes smooth degradation, but this remains a simulator based study rather than a complete characterization of device noise, calibration error, decoherence, or measurement drift on real hardware.

## 14 Further Experiments and Ablations

In this section, we further explore SQMM. All SQMM results use the model of Section $5$ at the epochs and growth intervals of the corresponding main table, unless stated otherwise.

## 14.1 Bands & Time

![](images/ba77c074ea15f69861c18b54a7d5f997fe441f991233d7b708084eea5c415d05.jpg)

![](images/0f3daea855ded05370b85d90819571c5552f2c28d0230af0740d4bb84022cc2b.jpg)  
Figure 4: Mixed state scaling ablation on Cello. Training time against the number of sequential bands (left) and the number of memory qubits (right). Each point is one run with 600 training epochs and a fixed depth per band.

The setup for Figure 4 matches the cello data protocol in Section 5.1. The models were run without spectral loss, without classical preprocessing, and with linear postprocessing for 600 epochs. Runs used five data reuploads per band. The goal was to isolate how much the quantum component captures on its own and how circuit width and band count affect wall time and MSE. Adding bands and memory qubits lowered the reconstruction error in these runs, at the training time cost shown in Figure 4. Results reaffirm the known simulator cost of added qubits, where each additional qubit doubles the state dimension and increases time and memory requirements as demonstrated in the right graph. Even a very small model with $q = 3 , m = 1$ , 11 bands, and 627 quantum parameters can take almost an hour to train on 1000 coordinates. Bands in the 5 to 7 range gave the best tradeoff among runtime, parameter count, and reconstruction quality, which guided the later model search toward architectures with fewer than 7 bands.

## 14.2 Noise & Hardware

We perturb every gate angle with Gaussian noise of standard deviation σ during training and evaluation (Table 8). The error grows smoothly with σ on the images and from $\sigma = 0 . 0 3$ on Cello, and the images are more sensitive to gate noise than Cello. Up to $\sigma = 0 . 0 3$ SQMM stays below every evaluated baseline on all four samples, and on Cello it does so at every σ. Figure 5 shows the reconstructions at every σ.

Table 8: Gaussian gate angle noise with SD σ in training and evaluation of the final model, MSE↓ $( \times 1 0 ^ { - 3 } )$ , mean ± SD. σ = 0 is the noiseless model.
<table><tr><td>Perturbation</td><td>Cello</td><td>Astronaut Camera</td><td></td><td>Coffee</td></tr><tr><td>σ = 0</td><td> $0 . 4 8 { \pm } 0 . 0 8$ </td><td> $1 . 1 2 { \pm } 0 . 1 0$ </td><td> $0 . 3 8 { \pm } 0 . 0 8$ </td><td> $0 . 7 8 { \pm } 0 . 0 5$ </td></tr><tr><td> $\sigma = 0 . 0 1$ </td><td> $0 . 4 6 { \pm } 0 . 1 1$ </td><td> $1 . 4 6 { \pm } 0 . 1 1$ </td><td> $0 . 4 4 \pm 0 . 0 5$ </td><td> $0 . 9 7 { \scriptstyle \pm 0 . 3 2 }$ </td></tr><tr><td> $\sigma = 0 . 0 3$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 . 6 2 { \pm } 0 . 1 4$ </td><td> $0 . 6 8 { \pm } 0 . 0 5$ </td><td> $1 . 1 0 { \pm } 0 . 1 6$ </td></tr><tr><td>σ = 0.1</td><td> $0 . 7 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $2 . 4 8 { \pm } 0 . 5 6 $ </td><td> $1 . 0 3 { \pm } 0 . 1 5$ </td><td> $1 . 8 0 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td> $\sigma = 0 . 3$ </td><td> $1 . 1 3 { \pm } 0 . 1 9$ </td><td> $4 . 8 9 \pm 0 . 5 8$ </td><td> $2 . 3 8 { \pm } 0 . 7 6$ </td><td> $3 . 9 0 { \scriptstyle \pm 1 . 0 0 }$ </td></tr></table>

![](images/58ed06fef231b3ea2855869e3f32cef7382a140400a0033decf2b837d150587b.jpg)  
Figure 5: Reconstructions under gate angle noise of SD σ in training and evaluation. Cello shows samples 400 to 600 with the target in grey and the prediction in red.

## 14.3 Loss Extension

1D rFFT. For 1D uniformly sampled audio, Eq. (15) with $\varepsilon = 0$ and $\Omega _ { \mathrm { g r i d } }$ the nonnegative discrete Fourier frequencies equals the real FFT scaled by $1 / T ,$ since the columns of Φ are orthogonal. The feature and probe traces are real valued, so the negative frequency coefficients are the conjugates of the positive ones and carry no further information. The 1D real FFT is therefore an efficient implementation of the finite grid spectral projection on this grid.

Final output spectral control. The ablation in Table 9 also uses a control loss that applies the same spectral comparison directly to the final prediction. This loss is not the proposed memory alignment objective. It does not inspect $\sigma _ { M } ^ { ( i ) } ( x )$ , does not use prefix occupancies, and does not provide an intermediate target for the memory lane. It only asks the final predicted signal to have the same normalized spectral energy distribution as the target signal on a finite frequency grid. For a generic nonnegative spectrum R on a finite grid Ω, write this normalization compactly as

$$
\mathcal { N } _ { \delta } R ( \omega ) = \frac { R ( \omega ) + \delta } { \sum _ { \omega ^ { \prime } \in \Omega } R ( \omega ^ { \prime } ) + | \Omega | \delta } .\tag{65}
$$

Let $x _ { 1 } , \ldots , x _ { T }$ be the sampled coordinates used in a training batch, and let

$$
\hat { y } _ { t } = \hat { y } ( x _ { t } ) , \qquad y _ { t } = y ( x _ { t } ) .
$$

For each output channel $o \in \{ 1 , \ldots , d _ { y } \}$ , estimate Fourier coefficients on a grid $\Omega _ { \mathrm { o u t } }$ using the same FFT, DFT, or least squares estimator as above. Write these coefficients as

$$
\widehat { a } _ { \widehat { y } , o } ( \omega ) , \qquad \widehat { a } _ { y , o } ( \omega ) , \qquad \omega \in \Omega _ { \mathrm { o u t } } .
$$

The final output spectral energies are

$$
P _ { \hat { y } } ^ { \mathrm { o u t } } ( \omega ) = \sum _ { o = 1 } ^ { d _ { y } } \left| \widehat { a } _ { \hat { y } , o } ( \omega ) \right| ^ { 2 } , \qquad P _ { y } ^ { \mathrm { o u t } } ( \omega ) = \sum _ { o = 1 } ^ { d _ { y } } \left| \widehat { a } _ { y , o } ( \omega ) \right| ^ { 2 } .\tag{66}
$$

After applying the same normalization operator as Eq. (18), the output spectral control loss is

$$
\mathcal { L } _ { \mathrm { o u t } } = \sum _ { \omega \in \Omega _ { \mathrm { o u t } } } \left( \mathcal { N } _ { \delta } P _ { \hat { y } } ^ { \mathrm { o u t } } ( \omega ) - \mathrm { s g } \big [ \mathcal { N } _ { \delta } P _ { y } ^ { \mathrm { o u t } } ( \omega ) \big ] \right) ^ { 2 } .\tag{67}
$$

The corresponding control objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { o u t } } \mathcal { L } _ { \mathrm { o u t } } . } \end{array}\tag{68}
$$

The stop gradient is applied only to the target spectrum. Gradients flow through the final prediction $\hat { y } ,$ so this control regularizes the predictor output itself. In contrast, Eq. (19) shapes the propagated memory spectrum after each prefix and leaves the supervised task loss responsible for fitting the final signal values. In the experiments, $\mathcal { L } _ { \mathrm { o u t } } { \mathrm { u s e s } } \lambda _ { \mathrm { o u t } } = 0 . 1$

No memory controls. The no memory control of Table 2 keeps the classical part of SQMM, including the projection that prepares the memory, and replaces the circuit by independent bands of four qubits. In every band the two data qubits are uploaded in every layer and read out through the same Pauli Z and ZZ features as in SQMM. The two further qubits are prepared once from the memory projection, take part in every entangling layer and are never measured. They start from |0 in every band, so nothing passes between bands. The bands use [1, 2, 3, 3, 4] layers, which gives 650 parameters against 642 for SQMM, and the control is trained with the protocol of Table 1 without an auxiliary loss, since it has no shared memory to align. Table 9 repeats this control on all four samples as No memory, local qubits, with 662 parameters against 654 for SQMM on the images, and uses two further controls. No memory qubits removes the memory qubits and their couplings, leaving two qubits per band. No memory qubits with a wider decoder widens the decoder to 11 hidden units, which restores the parameter count but not the qubit count.

Mechanism ablation. Table 9 extends the mechanism ablation of Table 2 to all four samples. Removing the memory qubits gives the largest error on every sample, and a wider decoder at a similar parameter count recovers only part of the gap. The control with local qubits has a higher error than SQMM on Cello, Astronaut and Camera and the same error on Coffee. Read only memory lies between no memory and read and write memory. ${ \mathcal { L } } _ { \mathrm { o u t } }$ leaves the error of the memory model nearly unchanged, while $\dot { \mathcal { L } } _ { \mathrm { a l i g n } }$ gives the lowest error on every sample. The one memory qubit settings train SQMM with $\mathcal { L } _ { \mathrm { a l i g n } }$ and a single memory qubit. The first keeps the bands of SQMM, and the second adds one reupload to band 1 and two to each of bands 2 to 5, which matches the parameter count of SQMM. With one memory qubit the error rises on every sample. At the parameter count of SQMM the added reuploads close part of the gap, most on Cello where the two errors lie within one SD, and SQMM keeps the lowest error on every sample. The second memory qubit is therefore worth more than the same number of parameters spent on reuploads.

Table 9: Mechanism and control ablation on the classic benchmark, MSE↓ $( \times 1 0 ^ { - 3 } )$ $\mathrm { m e a n } \pm \mathrm { S D }$ Params give the live parameters of each setting. No memory qubits removes the memory qubits and their couplings, and No memory, local qubits is the control of Table 2. Both one qubit memory rows use $\mathcal { L } _ { \mathrm { a l i g n } }$ , and the added reuploads match the parameter count of SQMM. Best in bold.
<table><tr><td>Setting</td><td></td><td>Params 1D Params 2D</td><td>Cello</td><td>Astronaut Camera</td><td></td><td>Coffee</td></tr><tr><td>No memory qubits</td><td>534</td><td>544</td><td> $0 . 8 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $2 . 9 4 \pm 0 . 5 4$ </td><td> $0 . 8 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 . 5 9 { \pm } 0 . 4 2 $ </td></tr><tr><td>No memory qubits, wider decoder</td><td>636</td><td>646</td><td> $0 . 6 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 . 9 4 \pm 0 . 1 8$ </td><td> $0 . 7 3 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 . 3 3 { \pm } 0 . 2 1 $ </td></tr><tr><td>No memory, local qubits</td><td>650</td><td>662</td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 . 4 1 { \pm } 0 . 1 8$ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 8 0 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td>Read only memory</td><td>626</td><td>638</td><td> $0 . 6 1 { \pm } 0 . 1 3$ </td><td> $1 . 5 2 { \pm } 0 . 0 9$ </td><td> $0 . 4 8 { \pm } 0 . 0 9$ </td><td> $1 . 0 0 { \pm } 0 . 2 1 $ </td></tr><tr><td>Read and write memory</td><td>642</td><td>654</td><td> $0 . 5 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 . 2 3 { \pm } 0 . 2 1 $ </td><td> $0 . 4 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 8 8 { \pm } 0 . 1 1$ </td></tr><tr><td>Memory with  ${ \mathcal { L } } _ { \mathrm { o u t } }$ </td><td>642</td><td>654</td><td> $0 . 5 4 \pm 0 . 0 6$ </td><td> $1 . 2 0 { \pm } 0 . 1 5$ </td><td> $0 . 4 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 8 8 { \pm } 0 . 0 8$ </td></tr><tr><td>One qubit memory with  $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>588</td><td>599</td><td> $0 . 6 6 { \pm } 0 . 0 9$ </td><td> $1 . 7 8 { \pm } 0 . 2 0 $ </td><td> $0 . 5 9 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 . 3 4 \pm 0 . 1 1$ </td></tr><tr><td>One qubit memory, more reuploads</td><td>642</td><td>653</td><td> $0 . 5 2 { \pm } 0 . 0 5$ </td><td> $1 . 3 8 { \pm } 0 . 1 7$ </td><td> $0 . 4 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $1 . 2 1 { \pm } 0 . 2 1 $ </td></tr><tr><td>Memory with  $\mathcal { L } _ { \mathrm { a l i g n } }$  (Ours)</td><td>642</td><td>654</td><td> $\mathbf { 0 . 4 8 \pm 0 . 0 8 }$ </td><td> $\mathbf { 1 . 1 2 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 3 8 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $\mathbf { 0 . 7 8 \pm 0 . 0 5 }$ </td></tr></table>

Memory switched off at evaluation. To test whether the trained model uses its memory, we take the five runs of Table 1 and evaluate each again with every memory read and write coupling set to zero. The read and write gates are controlled rotations and $X Y$ gates, which reduce to the identity at zero angle, so the bands run exactly as trained while no information passes through the memory register. Nothing is retrained. Switching the memory off raises the error by a factor of 36 on Cello, 388 on Astronaut, 291 on Camera and 149 on Coffee (Table 10), far above the error of a model trained without memory. The trained bands therefore rely on what the memory carries, and the memory register is not a component that the model learns to ignore.

Table 10: Memory switched off at evaluation on the classic benchmark, MSE↓ $( \times 1 0 ^ { - 3 } )$ $\mathrm { m e a n } \pm \mathrm { S D }$ over the five runs of Table 1. The first two rows evaluate the same trained models, with every read and write coupling set to zero in the second. The last row is the model without memory qubits of Table 9. <sup>⋆</sup> marks our model.
<table><tr><td>Setting</td><td>Cello</td><td>Astronaut</td><td>Camera</td><td>Coffee</td></tr><tr><td>SQMM*, memory on</td><td> $0 . 4 8 { \pm } 0 . 0 8$ </td><td> $1 . 1 2 { \pm } 0 . 1 0$ </td><td> $0 . 3 8 { \pm } 0 . 0 8$ </td><td> $0 . 7 8 { \pm } 0 . 0 5$ </td></tr><tr><td>Memory switched off at evaluation</td><td> $1 7 . 0 5 { \pm } 4 . 6 5$ </td><td> $4 4 2 . 4 0 { \scriptstyle \pm 1 9 4 . 6 7 }$ </td><td> $1 1 0 . 8 4 \pm 3 7 . 9 8$ </td><td> $1 1 3 . 6 9 { \scriptstyle \pm 5 0 . 6 5 }$ </td></tr><tr><td>Trained without memory qubits</td><td> $0 . 8 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $2 . 9 4 \pm 0 . 5 4$ </td><td> $0 . 8 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 . 5 9 { \pm } 0 . 4 2 $ </td></tr></table>

Memory anatomy. Memory is an input dependent reduced quantum state rather than a feature vector, so we resolve it at every band boundary along three axes in Table 11. Spectral alignment compares the memory occupancy spectrum with the accumulated prefix spectrum by cosine similarity and Jensen Shannon divergence. Coordinate spread is the variance of $\langle P \rangle ( x )$ across coordinates summed over the 15 non identity Paulis, which measures how far the state moves with the input. Write innovation is the norm of the Bloch vector that a write produces from a maximally mixed input, which isolates what the write injects from what memory retains. With $\mathcal { L } _ { \mathrm { a l i g n } }$ , cosine agreement rises from the first to the last boundary while the coordinate spread decreases with depth. The write innovation is largest at the first boundary. Switching the memory off at evaluation raises the error of the trained Astronaut model by a factor of 388 (Table 10).

Table 11: Memory anatomy on Astronaut at each band boundary with $\mathcal { L } _ { \mathrm { a l i g n } }$
<table><tr><td>Boundary</td><td>Cosine↑ JS↓</td><td>Coordinate spread</td><td>Write innovation</td></tr><tr><td>1</td><td>0.75 0.12</td><td>1.64</td><td>0.21</td></tr><tr><td>2</td><td>0.82 0.11</td><td>1.26</td><td>0.15</td></tr><tr><td>3</td><td>0.82 0.14</td><td>1.02</td><td>0.08</td></tr><tr><td>4</td><td>0.82 0.17</td><td>0.80</td><td>0.09</td></tr></table>

## 14.4 Finite shot measurement

Table 12 trains and evaluates SQMM with every expectation estimated from S shots, on all four samples at every S. On every sample the error decreases with S, and on Cello at 2048 shots it is 59.95 % above exact evaluation. The images need more shots than Cello to approach exact evaluation. On Cello SQMM is below every evaluated baseline from 32 shots on, while on the images its error at 2048 shots is still above that of the strongest baselines. Figure 6 shows how the reconstructions change with the shot count.

Table 12: MSE↓ $( \times 1 0 ^ { - 3 } )$ with expectations estimated from S shots in training and evaluation, mean $\pm \mathrm { s D }$
<table><tr><td>Shots S</td><td>Cello</td><td>Astronaut</td><td>Camera</td><td>Coffee</td></tr><tr><td>16</td><td> $3 . 6 5 { \pm } 0 . 2 3 $ </td><td> $1 8 . 3 9 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $7 . 6 4 \pm 0 . 7 8$ </td><td> $1 0 . 0 4 { \scriptstyle \pm 0 . 4 5 }$ </td></tr><tr><td>32</td><td> $3 . 0 5 { \pm } 0 . 1 2$ </td><td> $1 2 . 2 4 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $5 . 8 9 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $8 . 1 5 { \pm } 1 . 0 1$ </td></tr><tr><td>64</td><td> $2 . 3 0 { \pm } 0 . 1 7$ </td><td> $9 . 5 0 { \scriptstyle \pm 0 . 7 0 } $ </td><td> $4 . 0 7 \pm 0 . 2 9$ </td><td> $5 . 8 2 \pm 0 . 4 3$ </td></tr><tr><td>128</td><td> $1 . 8 1 { \pm } 0 . 1 4$ </td><td> $7 . 8 6 \pm 0 . 4 3$ </td><td> $3 . 3 8 { \pm } 0 . 2 6$ </td><td> $4 . 6 1 \pm 0 . 5 7$ </td></tr><tr><td>256</td><td> $1 . 3 9 { \pm } 0 . 0 8$ </td><td> $6 . 1 7 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $2 . 6 3 { \pm } 0 . 4 4$ </td><td> $3 . 7 6 { \pm } 1 . 1 4 $ </td></tr><tr><td>512</td><td> $1 . 1 2 { \pm } 0 . 0 7$ </td><td> $4 . 7 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 . 2 1 { \pm } 0 . 3 8$ </td><td> $3 . 0 0 { \pm } 0 . 3 5 $ </td></tr><tr><td>1024</td><td> $0 . 9 6 { \pm } 0 . 0 9$ </td><td> $3 . 7 3 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $1 . 7 0 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $2 . 2 2 { \pm } 0 . 2 0 $ </td></tr><tr><td>2048</td><td> $0 . 7 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $2 . 8 6 { \pm } 0 . 5 3 $ </td><td> $1 . 2 7 { \pm } 0 . 1 6$ </td><td> $1 . 7 9 2 0 . 1 6$ </td></tr><tr><td>exact</td><td> $0 . 4 8 { \pm } 0 . 0 8$ </td><td> $1 . 1 2 { \pm } 0 . 1 0$ </td><td> $0 . 3 8 { \pm } 0 . 0 8$ </td><td> $0 . 7 8 { \pm } 0 . 0 5$ </td></tr></table>

![](images/7abeec67883b0f1e5c3842d39aba4f58638962fbc3e52ec5478c8860b72b349c.jpg)  
Figure 6: Reconstructions with expectations estimated from S shots in training and evaluation. Cello shows samples 400 to 600 with the target in grey and the prediction in red.

## 14.5 Model configurations

All experiments run on CPU nodes of a shared internal cluster, and the CPU type and memory of a node vary with the availability of the shared system. Table 13 lists the settings of SQMM and Table 14 the configurations of the classical baselines. The same SQMM is used for every dataset, with the coordinate encoding and spectrum bins of the input dimension and three outputs for RGB images.

Table 13: Settings of SQMM.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Bands Encoding</td><td>5 bands of 2 data qubits with 2, 3, 4, 4 and 5 uploads the raw coordinate and 16 dyadic sinusoidal frequencies uploaded through  $R _ { Y }$ </td></tr><tr><td></td><td>in 1D, the raw coordinates and 8 consecutive frequencies per axis uploaded through 1  $R  { \gamma }$  and  $R x$  with a shared bias in 2D 2</td></tr><tr><td>Memory</td><td>2 qubits coupled to every data qubit, no entangler of its own, initialized by  $R _ { Y }$  and  $R _ { X }$  uploads of the encoding</td></tr><tr><td>Read</td><td>controlled  $R  { \boldsymbol { Y } }$  before the first upload of every band, an extra controlled  $R x$  in</td></tr><tr><td>Write</td><td>band 5  $X Y$  couplings in bands 1 to 4, none in band 5</td></tr><tr><td>Readout</td><td>Pauli  $Z$  and  $\bar { Z } Z$  expectations of every band, concatenated</td></tr><tr><td>Decoder Training</td><td>MLP with one hidden layer of width 5 and GELU, three outputs for RGB images</td></tr><tr><td>Band growth</td><td>Adam, learning rate 0.03, full batch, one update per epoch one band per growth interval of 25 updates for 300 epochs and 50 updates for</td></tr><tr><td> $\mathcal { L } _ { \mathrm { a l i g n } }$  Bins</td><td>600 epochs, with a fade in of 5 updates  $\lambda _ { \mathrm { a l i g n } } = 3 , \delta = 1 0 ^ { - 8 }$  , boundaries 1 to 3, switched off after four growth intervals all  $\lfloor { T } / 2 \rfloor + 1$  nonnegative real FFT bins in 1D (501 on Cello, 513 on the audio clips), the  $3 2 \times 3 2$  lowest signed frequencies of the grid FFT in 2D</td></tr></table>

Spectral memory alignment. The feature spectrum uses the Pauli $Z$ and ZZ features of every active band, and the memory spectrum uses the $4 ^ { m }$ full Pauli signals $\chi _ { i , s } ^ { M } ( x )$ of the memory state after band i, which are available in simulation. All coordinates lie on regular grids, so Eq. (51) is used with $\varepsilon = 0$ and DFT frequencies, where it equals the discrete Fourier transform scaled by $1 / T$ . The implementation uses the orthonormal FFT, which multiplies both spectra by T. Energies are summed over the feature components or Pauli words of a band. The prefix target $P _ { i } ^ { \check { F } }$ is the cumulative sum of the band energies of bands 1 to i, normalized after summation, and $P _ { i } ^ { M }$ is normalized in the same way, so the factors $2 ^ { - m }$ and $T$ cancel up to a rescaling of δ to $\delta / T$ for the feature spectrum and to $\delta / ( 2 ^ { m } T )$ for the memory spectrum. $\mathcal { L } _ { \mathrm { a l i g n } }$ is the squared difference of the normalized spectra averaged over the active boundaries and bins, without frequency weighting. Boundary i enters the loss when band i + 1 is added, and the loss is switched off when the fifth band is added, so boundary 4, which is read only by the last band, is not supervised. The target is detached, and gradients of the memory side reach every parameter the memory state depends on.

Classical baselines. Every classical baseline has one configuration for audio (1D), fitted on Cello, GTZAN, UrbanSound8K and VCTK, and one for images (2D), fitted on Astronaut, Camera, Coffee, CIFAR10, Kodak and DIV2K with three outputs for RGB. A configuration fixes the hidden width, the number of hidden layers and the learning rate. The 1D shape is the shape selected for the audio comparison and the 2D shape the shape selected for the CIFAR10 comparison, both from every width and depth shape whose parameter count falls inside the budget. The learning rate is selected from $[ 3 \cdot 1 0 ^ { - 2 } , 1 0 ^ { - 3 } ]$ , by the lowest geometric mean of the best epoch MSE over Cello and the audio corpora in 1D and over Astronaut, Camera, Coffee and CIFAR10 in 2D. Kodak and DIV2K reuse the selected 2D configuration. All baselines are trained with Adam, full batch updates and no learning rate schedule on the MSE of targets in [−1, 1], for the epochs of the corresponding SQMM experiment, and are evaluated with their best epoch weights.

Every other setting keeps the default of the official implementation. Two baselines change a default. MSA-INR uses a signed input domain and, in 2D, a frequency comb rescaled to the Nyquist band of the 32 × 32 grid. The published CAFE encoder of 88 random Fourier features and Chebyshev order 30 admits a single hidden unit within the budget, so we also report CAFE (enc.), which scales the encoder down to 5 features and order 2 in 1D and 21 features and order 7 in 2D.

Quantum baselines. QIREN uses its published hybrid architecture of width 8 with 2 hybrid layers and CZ entanglers, with spectrum layer 2 (649 parameters in 1D, 657 in 2D), and QIREN+ uses spectrum layer 3 (793 in 1D, 801 in 2D), and learning rate 0.03. QVF uses its published 671 parameter image model with 6 embedding frequencies, a hidden width of 10, one repetition and learning rate 0.01.

Table 14: Configurations of the classical baselines. Shape is hidden width × hidden layers. TUNER has no 1D configuration since its initialization requires 2D coordinates.
<table><tr><td></td><td colspan="2">1D (audio)</td><td colspan="2">2D (images)</td></tr><tr><td>Model</td><td>Shape</td><td>lr</td><td>Shape</td><td>lr</td></tr><tr><td>FINER</td><td> $1 0 \times 6$ </td><td> $1 0 ^ { - 3 }$ </td><td> $2 4 \times 1 3 \cdot 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>SIREN</td><td> $1 4 \times 3$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 4 \times 3 ~ 3 \cdot 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>NeRF</td><td> $1 7 \times 1$ </td><td> $1 0 ^ { - 2 }$ </td><td> $1 2 \times 3 ~ 3 \cdot 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>KAN</td><td>2 × 4</td><td> $3 \cdot 1 0 ^ { - 2 }$ </td><td>2 × 4</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>WIRE</td><td>25 × 2 3 · 10−3</td><td></td><td> $2 5 \times 2 ~ 3 \cdot 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>R+RFF</td><td>10 × 5 3 · 10−3</td><td></td><td>24 × 1 3· 10−2</td><td></td></tr><tr><td>ReLU</td><td></td><td>11 × 4 3· 10−3</td><td>11 × 4</td><td>10⁻²</td></tr><tr><td>Tanh</td><td></td><td>16 × 2 3 · 10−2</td><td> $1 1 \times 4 3 \cdot 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>IGA</td><td>14 × 3</td><td> $1 0 ^ { - 3 }$ </td><td>14 × 3 3 · 10−3</td><td></td></tr><tr><td>FLAIR</td><td></td><td> $1 0 \times 4 3 \cdot 1 0 ^ { - 3 }$ </td><td> $1 7 \times 2 3 \cdot 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>TUNER</td><td></td><td></td><td> $2 5 \times 1 3 \cdot 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>VI3NR</td><td></td><td> $1 0 \times 6 3 \cdot 1 0 ^ { - 2 }$ </td><td>10 × 6 3 · 10−2</td><td></td></tr><tr><td>MSA-INR</td><td></td><td> $2 4 \times 1 3 \cdot 1 0 ^ { - 2 }$ </td><td> $2 4 \times 1 3 \cdot 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>CAFE (enc.)</td><td></td><td> $1 3 \times 1 3 \cdot 1 0 ^ { - 2 }$ </td><td>4 × 1 2·10−2</td><td></td></tr><tr><td>CAFE</td><td></td><td>1 ×1 2·10−2</td><td>1 × 1</td><td>10⁻2</td></tr></table>

## 14.6 Base benchmark best runs

Table 15 reports the base benchmark of Table 1 in the best run convention of QIREN, with all evaluated baselines and QIREN+, which adds one spectrum layer to QIREN at a larger parameter count. SQMM has the lowest best run error on every sample.

Table 15: Classic INR benchmark in the best run convention of QIREN, MSE↓ $( \times 1 0 ^ { - 3 } )$ of the best single run. The SQMM rows take the best of the five runs in Table 1. QIREN+ is QIREN with one extra spectrum layer. <sup>⋆</sup> marks our model. Best in bold.
<table><tr><td>Model</td><td>Params 1D</td><td>Params 2D</td><td>Cello</td><td>Astronaut</td><td>Camera</td><td>Coffee</td></tr><tr><td>SQMM*</td><td>642</td><td>654</td><td>0.38</td><td>1.01</td><td>0.28</td><td>0.73</td></tr><tr><td>SQMM2048</td><td>642</td><td>654</td><td>0.71</td><td>2.36</td><td>1.09</td><td>1.57</td></tr><tr><td>QIREN</td><td>649</td><td>657</td><td>6.25</td><td>5.34</td><td>1.80</td><td>1.73</td></tr><tr><td>QVF</td><td></td><td>671</td><td></td><td>11.00</td><td>3.22</td><td>7.76</td></tr><tr><td>QIREN+</td><td>793</td><td>801</td><td>5.66</td><td>3.17</td><td>1.39</td><td>1.31</td></tr><tr><td>FINER</td><td>691</td><td>697</td><td>3.32</td><td>2.49</td><td>1.25</td><td>1.54</td></tr><tr><td>SIREN</td><td>673</td><td>687</td><td>4.93</td><td>1.82</td><td>0.44</td><td>1.96</td></tr><tr><td>NeRF</td><td>698</td><td>997</td><td>4.42</td><td>4.52</td><td>1.16</td><td>2.82</td></tr><tr><td>KAN</td><td>702</td><td>1222</td><td>3.07</td><td>43.40</td><td>21.02</td><td>21.10</td></tr><tr><td>WIRE</td><td>1294</td><td>1311</td><td>4.81</td><td>19.19</td><td>26.41</td><td>23.89</td></tr><tr><td>R+RFF</td><td>661</td><td>673</td><td>5.45</td><td>4.23</td><td>2.50</td><td>3.61</td></tr><tr><td>ReLU</td><td>672</td><td>683</td><td>7.65</td><td>11.36</td><td>3.15</td><td>4.21</td></tr><tr><td>Tanh</td><td>689</td><td>683</td><td>14.99</td><td>17.79</td><td>7.32</td><td>12.56</td></tr><tr><td>IGA</td><td>673</td><td>687</td><td>4.84</td><td>1.38</td><td>0.59</td><td>1.11</td></tr><tr><td>FLAIR</td><td>703</td><td>687</td><td>4.69</td><td>4.40</td><td>2.69</td><td>2.64</td></tr><tr><td>TUNER</td><td></td><td>701</td><td></td><td>4.73</td><td>5.38</td><td>3.78</td></tr><tr><td>VI3NR</td><td>691</td><td>701</td><td>7.78</td><td>6.24</td><td>3.22</td><td>3.76</td></tr><tr><td>MSA-INR</td><td>673</td><td>697</td><td>6.37</td><td>1.58</td><td>0.99</td><td>1.08</td></tr><tr><td>CAFE (enc.)</td><td>703</td><td>709</td><td>2.81</td><td>3.42</td><td>1.01</td><td>3.50</td></tr><tr><td>CAFE</td><td>625</td><td>715</td><td>11.52</td><td>30.97</td><td>36.94</td><td>19.73</td></tr></table>

## 14.7 Audio results

Table 16 compares all models on the three audio corpora. SQMM has the lowest error on every corpus. FINER and CAFE (enc.) are the closest baselines, while QIREN, QIREN+ and MSA-INR stay close to an order of magnitude above SQMM. Table 17 reports further SQMM metrics. Speech

(VCTK) is reconstructed with the lowest MSE and the highest PSNR, and music (GTZAN) has the highest log spectral distance.  
Table 16: Audio reconstruction MSE↓ $( \times 1 0 ^ { - 3 } )$ for all models, $\mathrm { m e a n } \pm \mathrm { S D } .$ . QIREN+ adds one spectrum layer to QIREN. <sup>⋆</sup> marks our model. Best in bold.
<table><tr><td>Model</td><td>Params</td><td>GTZAN</td><td>US8K</td><td>VCTK</td></tr><tr><td>SQMM*</td><td>642</td><td> $\mathbf { 4 . 7 1 \pm 0 . 1 7 }$ </td><td> ${ \bf 6 . 1 1 \pm 0 . 2 7 }$ </td><td> $\mathbf { 1 . 0 0 \pm 0 . 0 3 }$ </td></tr><tr><td>QIREN+</td><td>793</td><td> $4 0 . 9 2 { \pm } 1 . 1 5$ </td><td> $5 0 . 1 8 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $9 . 7 3 { \scriptstyle \pm 0 . 7 6 }$ </td></tr><tr><td>QIREN</td><td>649</td><td> $4 4 . 1 9 { \scriptstyle \pm 1 . 8 8 }$ </td><td> $5 1 . 8 3 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $1 0 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>FINER</td><td>691</td><td> $5 . 1 4 \pm 1 . 0 6$ </td><td> $8 . 5 6 \pm 2 . 3 5$ </td><td> $1 . 6 6 { \pm } 0 . 2 1 $ </td></tr><tr><td>SIREN</td><td>673</td><td> $8 . 6 5 { \pm } 0 . 6 1$ </td><td> $1 5 . 5 5 { \pm } 1 . 4 9$ </td><td> $2 . 9 8 { \pm } 0 . 1 5$ </td></tr><tr><td> $\mathrm { N e R F }$ </td><td>698</td><td> $8 . 8 9 \pm 0 . 4 4$ </td><td> $1 3 . 9 6 { \pm } 0 . 7 2$ </td><td> $2 . 5 9 { \pm } 0 . 1 5$ </td></tr><tr><td> $\mathrm { K A N }$ </td><td>702</td><td> $1 7 . 2 6 { \pm } 0 . 3 7$ </td><td> $1 8 . 4 8 { \pm } 0 . 3 8 $ </td><td> $3 . 7 5 { \pm } 0 . 1 2$ </td></tr><tr><td> $\mathrm { W I R E }$ </td><td>1294</td><td> $2 5 . 6 4 \pm 1 2 . 7 8$ </td><td> $3 4 . 3 7 { \pm } 7 . 3 1$ </td><td> $5 . 5 1 { \pm } 2 . 7 3 $ </td></tr><tr><td> $\mathrm { R } { + } \mathrm { R F } \mathrm { F }$ </td><td>661</td><td> $3 7 . 5 3 { \pm } 7 . 3 0 $ </td><td> $4 8 . 5 0 { \scriptstyle \pm 4 . 4 0 }$ </td><td> $7 . 4 7 { \pm } 1 . 5 4 $ </td></tr><tr><td> $\mathrm { R e L U }$ </td><td>672</td><td> $5 8 . 9 0 { \scriptstyle \pm 4 . 7 7 }$ </td><td> $5 8 . 1 6 { \pm } 1 . 3 7 $ </td><td> $1 1 . 4 1 { \pm } 0 . 7 7$ </td></tr><tr><td>Tanh</td><td>689</td><td> $9 2 . 3 5 { \pm } 0 . 3 5 $ </td><td> $7 0 . 1 4 \pm 0 . 7 3$ </td><td> $1 9 . 2 9 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>IGA</td><td>673</td><td> $8 . 4 5 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $1 5 . 4 7 \pm 1 . 2 2$ </td><td> $2 . 6 9 \pm 0 . 1 3$ </td></tr><tr><td>FLAIR</td><td>703</td><td> $2 5 . 0 3 { \scriptstyle \pm 3 . 9 3 }$ </td><td> $3 9 . 4 7 { \scriptstyle \pm 3 . 7 5 }$ </td><td> $4 . 8 7 \pm 0 . 5 4$ </td></tr><tr><td>VI3NR</td><td>691</td><td> $5 0 . 9 8 { \scriptstyle \pm 3 . 1 9 }$ </td><td> $5 6 . 5 2 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $1 1 . 1 4 { \pm } 0 . 4 2$ </td></tr><tr><td> $\mathbf { M S A - I N R }$ </td><td>673</td><td> $4 2 . 1 1 \pm 3 . 3 9$ </td><td> $5 2 . 2 1 { \pm } 1 . 5 5 $ </td><td> $8 . 5 1 { \pm } 0 . 9 0 $ </td></tr><tr><td>CAFE (enc.)</td><td>703</td><td> $6 . 4 2 \pm 1 . 3 4$ </td><td> $1 0 . 8 3 { \pm } 3 . 7 5$ </td><td> $2 . 2 5 { \pm } 0 . 3 4$ </td></tr><tr><td>CAFE</td><td>625</td><td> $3 2 . 7 5 { \scriptstyle \pm 3 4 . 0 2 }$ </td><td> $3 6 . 9 6 { \scriptstyle \pm 2 8 . 2 8 }$ </td><td> $2 8 . 7 5 { \scriptstyle \pm 3 9 . 9 9 }$ </td></tr></table>

Table 17: SQMM audio metrics per corpus, $\mathrm { m e a n } \pm \mathrm { S D } .$
<table><tr><td>Corpus</td><td colspan="3"> $\mathrm { M S E \downarrow ( \times 1 0 ^ { - 3 } ) }$  PSNR↑ (dB)  $\mathrm { L S D \downarrow }$ </td></tr><tr><td>GTZAN</td><td> $4 . 7 1 { \pm } 0 . 1 7$ </td><td> $2 4 . 5 5 { \pm } 0 . 1 7$ </td><td> $7 . 1 0 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>US8K</td><td> $6 . 1 1 \pm 0 . 2 7$ </td><td> $2 2 . 9 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $5 . 3 8 { \pm } 0 . 0 8$ </td></tr><tr><td>VCTK</td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $3 3 . 9 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $5 . 5 7 { \scriptstyle \pm 0 . 0 6 }$ </td></tr></table>

## 14.8 Kodak and DIV2K

Table 18 compares all models on the 24 Kodak images at native resolution, and Tables 19 and 20 give the resolution ladders. SQMM has the lowest error at every resolution on both datasets, with the fewest parameters. The error of SQMM and of most baselines grows with resolution, since the parameter budget is fixed while the number of pixels grows. MSA-INR is the closest baseline up to $6 4 ^ { 2 }$ , and SIREN or IGA at larger sizes, and the relative gap between SQMM and the closest baseline narrows as the resolution grows. All 2D configurations are selected on $3 2 \times 3 2$ images, and WIRE and TUNER degrade sharply from $6 4 ^ { 2 }$ on. In Figure 7 the remaining SQMM error concentrates on edges and fine textures, while smooth regions are reproduced closely.

Table 18: Kodak at native resolution $( 7 6 8 \times 5 1 2 ,$ 24 images) for all models, MSE $( \times 1 0 ^ { - 3 } )$ , mean ± SD. <sup>⋆</sup> marks our model. Best in bold.
<table><tr><td>Model</td><td colspan="2">Params  $\mathrm { M S E \downarrow ( \times 1 0 ^ { - 3 } ) }$ </td></tr><tr><td>SQMM*</td><td>666</td><td> $3 2 . 0 8 { \pm } 0 . 1 5$ </td></tr><tr><td>FINER</td><td>747</td><td> $3 9 . 1 3 { \pm } 0 . 5 9$ </td></tr><tr><td>SIREN</td><td>717</td><td> $3 4 . 4 4 { \scriptstyle \pm 0 . 6 1 }$ </td></tr><tr><td>NeRF</td><td>1023</td><td> $3 4 . 7 9 2 0 . 4 0$ </td></tr><tr><td>KAN</td><td>1274</td><td> $5 4 . 1 0 { \pm } 1 . 3 5 $ </td></tr><tr><td>WIRE</td><td>1383</td><td> $4 7 . 5 8 { \pm } 3 . 2 1 $ </td></tr><tr><td> $\mathrm { R } { + } \mathrm { R F } \mathrm { F }$ </td><td>723</td><td> $3 8 . 5 4 { \scriptstyle \pm 0 . 6 4 }$ </td></tr><tr><td>ReLU</td><td>707</td><td> $4 1 . 5 1 { \pm } 1 . 3 7$ </td></tr><tr><td>Tanh</td><td>707</td><td> $4 5 . 0 0 { \scriptstyle \pm 0 . 3 5 }$ </td></tr><tr><td>IGA</td><td>717</td><td> $3 4 . 0 4 { \scriptstyle \pm 0 . 5 6 }$ </td></tr><tr><td>FLAIR</td><td>723</td><td> $3 9 . 0 4 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>TUNER</td><td>753</td><td> $1 1 8 . 4 7 { \scriptstyle \pm 0 . 9 9 }$ </td></tr><tr><td>VI3NR</td><td>723</td><td> $4 1 . 7 3 { \scriptstyle \pm 1 . 0 0 }$ </td></tr><tr><td>MSA-INR</td><td>747</td><td> $3 5 . 6 4 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td>CAFE (enc.)</td><td>719</td><td> $4 8 . 4 5 { \scriptstyle \pm 2 . 0 9 }$ </td></tr><tr><td>CAFE</td><td>719</td><td> $6 3 . 6 4 { \scriptstyle \pm 0 . 5 1 }$ </td></tr></table>

Table 19: Kodak resolution ladder (24 images, RGB), MSE↓ $( \times 1 0 ^ { - 3 } )$ , $\mathrm { m e a n } \pm \mathrm { S D } . ^ { \star }$ marks our model. Best in bold.
<table><tr><td>Model</td><td>Params</td><td> $3 2 ^ { 2 }$ </td><td> $6 4 ^ { 2 }$ </td><td> $1 2 8 ^ { 2 }$ </td><td> $2 5 6 ^ { 2 }$ </td><td> $\mathrm { n a t i v e 7 6 8 } { \times } 5 1 2$ </td></tr><tr><td>SQMM*</td><td>666</td><td> $\mathbf { 1 . 5 8 \pm 0 . 1 0 }$ </td><td> ${ \pm } . 4 3 { \pm } 0 . 0 3$ </td><td> $\mathbf { 1 1 . 5 4 \bot 0 . 0 5 }$ </td><td> $\mathbf { 1 8 . 5 2 } \pm \mathbf { 0 . 0 4 }$ </td><td> $\mathbf { 3 2 . 0 8 { \pm 0 . 1 5 } }$ </td></tr><tr><td>FINER</td><td>747</td><td> $5 . 8 2 \pm 0 . 3 6$ </td><td> $1 0 . 1 8 { \pm } 0 . 2 6 $ </td><td> $1 7 . 3 2 { \pm } 0 . 2 4 $ </td><td> $2 5 . 0 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $3 9 . 1 3 { \pm } 0 . 5 9$ </td></tr><tr><td>SIREN</td><td>717</td><td> $2 . 7 0 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $6 . 8 9 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $1 3 . 1 8 { \pm } 0 . 4 7$ </td><td> $2 0 . 5 7 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $3 4 . 4 4 { \scriptstyle \pm 0 . 6 1 }$ </td></tr><tr><td>NeRF</td><td>1023</td><td> $3 . 8 0 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $8 . 7 6 { \pm } 0 . 6 8$ </td><td> $1 4 . 9 4 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $2 1 . 5 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $3 4 . 7 9 2 0 . 4 0$ </td></tr><tr><td>KAN</td><td>1274</td><td> $1 9 . 4 9 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $2 5 . 5 3 { \pm } 1 . 2 6$ </td><td> $3 4 . 6 9 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $4 2 . 1 0 { \scriptstyle \pm 2 . 4 1 }$ </td><td> $5 4 . 1 0 { \scriptstyle \pm 1 . 3 5 }$ </td></tr><tr><td>WIRE</td><td>1383</td><td> $1 9 . 2 7 { \pm } 1 . 6 8$ </td><td> $5 2 . 1 0 { \scriptstyle \pm 3 . 7 3 }$ </td><td> $5 3 . 4 6 { \scriptstyle \pm 9 . 7 3 }$ </td><td> $3 7 . 3 7 { \pm } 3 . 4 0 $ </td><td> $4 7 . 5 8 { \pm } 3 . 2 1 $ </td></tr><tr><td> $\mathrm { R } { + } \mathrm { R F } \mathrm { F }$ </td><td>723</td><td> $3 . 3 5 { \pm } 0 . 0 7$ </td><td> $9 . 4 7 { \pm } 0 . 1 8$ </td><td> $1 6 . 8 9 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $2 4 . 5 3 { \pm } 0 . 3 4$ </td><td> $3 8 . 5 4 { \scriptstyle \pm 0 . 6 4 }$ </td></tr><tr><td> $\mathrm { R e L U }$ </td><td>707</td><td> $5 . 4 6 { \pm } 0 . 7 5 $ </td><td> $1 2 . 1 1 { \pm } 0 . 9 5 $ </td><td> $1 9 . 8 1 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $2 7 . 4 4 \pm 1 . 0 7$ </td><td> $4 1 . 5 1 { \pm } 1 . 3 7$ </td></tr><tr><td>Tanh</td><td>707</td><td> $7 . 1 7 { \pm } 0 . 2 9$ </td><td> $1 4 . 4 2 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $2 2 . 6 8 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $3 0 . 4 6 { \pm } 0 . 4 5$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 0 . 3 5 }$ </td></tr><tr><td>IGA</td><td>717</td><td> $3 . 1 1 \pm 0 . 3 4$ </td><td> $7 . 0 6 \pm 0 . 2 5$ </td><td> $1 3 . 6 5 { \pm } 0 . 3 6$ </td><td> $2 0 . 9 4 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $3 4 . 0 4 { \scriptstyle \pm 0 . 5 6 }$ </td></tr><tr><td>FLAIR</td><td>723</td><td> $3 . 4 8 { \pm } 0 . 0 8$ </td><td> $9 . 5 7 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $1 7 . 5 7 { \pm } 0 . 2 8 $ </td><td> $2 5 . 3 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 9 . 0 4 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>TUNER</td><td>753</td><td> $1 1 . 3 1 { \pm } 0 . 2 4$ </td><td> $6 2 . 2 5 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $9 3 . 8 8 { \pm } 0 . 4 1 $ </td><td> $1 0 4 . 3 7 { \pm } 0 . 2 6 $ </td><td> $1 1 8 . 4 7 { \scriptstyle \pm 0 . 9 9 }$ </td></tr><tr><td>VI3NR</td><td>723</td><td> $5 . 9 1 { \pm } 1 . 0 1 $ </td><td> $1 2 . 6 3 { \pm } 1 . 5 1 $ </td><td> $2 0 . 5 9 { \scriptstyle \pm 1 . 9 4 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $4 1 . 7 3 { \scriptstyle \pm 1 . 0 0 }$ </td></tr><tr><td>MSA-INR</td><td>747</td><td> $1 . 8 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $6 . 8 4 \pm 0 . 0 6$ </td><td> $1 4 . 0 2 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $2 1 . 5 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $3 5 . 6 4 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td>CAFE (enc.)</td><td>719</td><td> $7 . 5 0 { \pm } 1 . 9 0 $ </td><td> $1 6 . 2 1 { \pm } 3 . 1 3$ </td><td> $2 6 . 8 2 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $3 5 . 0 3 { \pm } 2 . 6 0$ </td><td> $4 8 . 4 5 { \scriptstyle \pm 2 . 0 9 }$ </td></tr><tr><td>CAFE</td><td>719</td><td> $2 6 . 9 0 { \scriptstyle \pm 7 . 1 6 }$ </td><td> $4 1 . 9 6 { \pm } 1 2 . 8 7$ </td><td> $4 4 . 6 8 { \pm } 1 . 6 5$ </td><td> $5 2 . 4 7 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $6 3 . 6 4 { \scriptstyle \pm 0 . 5 1 }$ </td></tr></table>

Table 20: DIV2K resolution ladder (10 images, RGB), MSE↓ $( \times 1 0 ^ { - 3 } )$ $\mathrm { m e a n } \pm \mathrm { S D }$ . The largest size is the 512<sup>2</sup> prepared image. <sup>⋆</sup> marks our model. Best in bold.
<table><tr><td>Model</td><td>Params</td><td> $3 2 ^ { 2 }$ </td><td> $6 4 ^ { 2 }$ </td><td> $1 2 8 ^ { 2 }$ </td><td> $2 5 6 ^ { 2 }$ </td><td> $\mathrm { f u l l 5 1 2 ^ { 2 } }$ </td></tr><tr><td>SQMM*</td><td>666</td><td>3.13±0.08</td><td> $\mathbf { 1 1 . 1 3 \bot 0 . 0 6 }$ </td><td> $2 3 . 1 6 { \pm } 0 . 3 4$ </td><td> $\mathbf { 3 5 . 1 5 \pm 0 . 4 3 }$ </td><td> ${ \pm } 4 . 1 3 { \pm } 0 . 4 4$ </td></tr><tr><td>FINER</td><td>747</td><td> $8 . 5 9 { \pm } 0 . 5 6 $ </td><td> $1 9 . 1 6 { \pm } 0 . 3 5 $ </td><td> $3 2 . 7 5 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $4 5 . 7 1 { \scriptstyle \pm 0 . 6 5 }$ </td><td>65.18±0.65</td></tr><tr><td>SIREN</td><td>717</td><td> $5 . 2 2 \pm 0 . 7 4$ </td><td> $1 4 . 3 7 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $2 7 . 0 7 { \pm } 1 . 0 2 $ </td><td> $3 9 . 3 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $5 8 . 4 1 \pm 1 . 1 4 $ </td></tr><tr><td>NeRF</td><td>1023</td><td> $6 . 9 1 \pm 0 . 2 0$ </td><td> $1 8 . 5 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $3 0 . 8 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $4 2 . 6 8 { \pm } 1 . 2 6 $ </td><td> $6 1 . 0 8 { \pm } 0 . 5 3 $ </td></tr><tr><td>KAN</td><td>1274</td><td> $2 8 . 7 9 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $4 3 . 6 7 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $6 0 . 4 2 { \scriptstyle \pm 2 . 7 1 }$ </td><td> $7 4 . 6 2 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $9 3 . 9 9 { \scriptstyle \pm 2 . 1 1 }$ </td></tr><tr><td>WIRE</td><td>1383</td><td> $2 7 . 2 7 { \scriptstyle \pm 2 . 0 2 }$ </td><td> $8 1 . 3 5 { \scriptstyle \pm 4 . 2 7 }$ </td><td> $9 0 . 9 7 { \scriptstyle \pm 8 . 5 5 }$ </td><td> $6 6 . 3 1 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $8 0 . 3 9 { \scriptstyle \pm 2 . 9 1 }$ </td></tr><tr><td>R+RFF</td><td>723</td><td> $6 . 1 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $1 8 . 4 2 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $3 2 . 6 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $4 5 . 8 0 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $6 5 . 1 4 { \scriptstyle \pm 0 . 8 5 }$ </td></tr><tr><td>ReLU</td><td>707</td><td> $1 2 . 0 8 { \pm } 1 . 7 0 $ </td><td> $2 7 . 5 8 { \pm } 3 . 0 8$ </td><td> $4 2 . 6 6 { \scriptstyle \pm 2 . 6 6 }$ </td><td> $5 6 . 2 8 { \scriptstyle \pm 2 . 6 3 }$ </td><td> $7 5 . 5 0 { \scriptstyle \pm 3 . 1 0 }$ </td></tr><tr><td>Tanh</td><td>707</td><td> $1 6 . 1 2 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $3 2 . 7 8 { \pm } 1 . 1 9$ </td><td> $4 9 . 4 1 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $6 2 . 3 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $8 2 . 7 2 { \scriptstyle \pm 1 . 9 7 }$ </td></tr><tr><td>IGA</td><td>717</td><td> $5 . 3 4 \pm 0 . 5 5$ </td><td> $1 5 . 2 2 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $4 0 . 0 2 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $5 8 . 4 0 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>FLAIR</td><td>723</td><td> $7 . 0 6 { \pm } 0 . 1 3$ </td><td> $1 9 . 8 3 { \pm } 0 . 2 8 $ </td><td> $3 5 . 1 8 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $4 8 . 1 8 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $6 8 . 1 1 \pm 1 . 3 1$ </td></tr><tr><td>TUNER</td><td>753</td><td> $1 6 . 0 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 5 . 5 6 { \pm } 0 . 9 2 $ </td><td> $1 4 7 . 6 9 { \pm } 0 . 3 0 $ </td><td> $1 6 4 . 1 3 { \pm } 1 . 0 4 $ </td><td> $1 8 2 . 9 4 { \pm } 1 . 2 2 $ </td></tr><tr><td>VI3NR</td><td>723</td><td> $1 0 . 9 5 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $2 4 . 1 8 { \pm } 2 . 1 5$ </td><td> $3 9 . 2 0 { \scriptstyle \pm 1 . 8 8 }$ </td><td> $5 2 . 5 4 \pm 2 . 2 5$ </td><td> $7 2 . 2 2 { \pm } 2 . 3 8 $ </td></tr><tr><td>MSA-INR</td><td>747</td><td> $3 . 5 8 { \pm } 0 . 0 9$ </td><td> $1 3 . 6 6 { \pm } 0 . 1 1$ </td><td> $2 7 . 3 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $3 9 . 8 4 { \pm } 0 . 3 1 $ </td><td> $5 9 . 1 0 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>CAFE (enc.)</td><td>719</td><td> $1 1 . 4 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $3 3 . 8 1 \pm 4 . 7 4$ </td><td> $5 3 . 7 7 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $6 7 . 6 5 { \scriptstyle \pm 3 . 3 7 }$ </td><td> $8 7 . 9 8 { \pm } 3 . 1 8$ </td></tr><tr><td>CAFE</td><td>719</td><td> $4 1 . 5 6 { \scriptstyle \pm 8 . 7 6 }$ </td><td> $7 0 . 4 5 { \scriptstyle \pm 2 3 . 4 9 }$ </td><td> $7 4 . 6 0 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $8 8 . 8 1 \pm 2 . 4 2$ </td><td> $1 0 7 . 5 9 { \pm 2 . 5 9 }$ </td></tr></table>

![](images/bb068913a3351879530d3343755fb22e5bc2d5b4c014f7798610000cce023fdd.jpg)  
Figure 7: SQMM reconstructions of Kodak and DIV2K images at $1 2 8 \times 1 2 8 .$ . Columns show the ground truth, the SQMM reconstruction, and the absolute error.

## 14.9 Ansatz Diagram

Figure 8 draws one band of the trained Cello model gate by gate, in the read, upload and write order of Section 4.

![](images/c0ac22245a0b715f3177563e9cef1df08d20de72eac3903a87dfcd0e999f8baf.jpg)  
Figure 8: Band 3 of SQMM, recorded from the trained Cello model. Controlled $R _ { Y }$ reads from the memory qubits $m _ { 1 }$ , m<sub>2</sub> precede the four $R _ { Y }$ uploads on the data qubits $q _ { 1 } , q _ { 2 } .$ , each followed by an entangling layer, and $X \bar { Y }$ couplings write to the memory at the end of the band.

## 14.10 CIFAR10 image reconstruction

Table 21 compares all models on the 100 grayscale CIFAR10 images. SQMM is best on all three metrics. MSA-INR, IGA and SIREN are the closest baselines, with about twice the MSE and more than twice the LPIPS of SQMM. Figure 9 shows the images with their SQMM reconstructions, which stay visually close to the ground truth at this resolution.

Table 21: Grayscale CIFAR10 metrics for all models, mean $\pm \mathrm { \bf ~ S D }$ . MSE is on the $[ - 1 , 1 ]$ range, SSIM and LPIPS on unit range images. QIREN+ adds one spectrum layer to $\mathrm { Q I R E N . } ^ { \star }$ marks our model. Best in bold.
<table><tr><td>Model</td><td>Params SSIM↑</td><td> $( \times 1 0 ^ { - 2 } )$ </td><td> $\mathrm { L P I P S \downarrow } ( \times 1 0 ^ { - 4 } )$ </td><td> $\mathrm { M S E \downarrow ( \times 1 0 ^ { - 3 } ) }$ </td></tr><tr><td>SQMM*</td><td>654</td><td> $\mathbf { 9 9 . 2 1 } { \pm } \mathbf { 0 . 0 3 }$ </td><td> $\mathbf { 3 . 4 4 \pm 0 . 2 5 }$ </td><td> $\mathbf { 0 . 4 6 { \scriptstyle \pm 0 . 0 2 } }$ </td></tr><tr><td>QIREN+</td><td>801</td><td> $9 7 . 6 9 { \pm } 0 . 1 9$ </td><td> $3 0 . 4 1 \pm 2 . 9 8$ </td><td> $1 . 6 7 { \pm } 0 . 1 3$ </td></tr><tr><td>QVF</td><td>671</td><td> $9 2 . 9 0 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $1 0 4 . 8 4 { \pm } 8 . 3 8 $ </td><td> $5 . 1 2 { \pm } 0 . 1 7$ </td></tr><tr><td>FINER</td><td>697</td><td> $9 7 . 4 3 { \pm } 0 . 3 9$ </td><td> $1 4 . 1 7 { \scriptstyle \pm 3 . 6 9 }$ </td><td> $1 . 4 5 { \pm } 0 . 2 1 $ </td></tr><tr><td>SIREN</td><td>687</td><td> $9 8 . 3 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $8 . 2 2 \pm 1 . 8 9$ </td><td> $0 . 9 8 { \pm } 0 . 2 1$ </td></tr><tr><td>WIRE</td><td>1311</td><td> $8 3 . 2 1 { \pm } 2 . 5 1 $ </td><td> $3 0 6 . 7 9 { \scriptstyle \pm 6 2 . 9 1 }$ </td><td> $1 8 . 2 3 { \pm } 2 . 6 8 $ </td></tr><tr><td>R+RFF</td><td>673</td><td> $9 6 . 1 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $9 3 . 6 0 { \scriptstyle \pm 4 7 . 0 0 }$ </td><td> $2 . 7 8 { \pm } 0 . 6 2$ </td></tr><tr><td>ReLU</td><td>683</td><td> $9 3 . 0 1 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $1 9 5 . 1 5 { \pm } 2 0 . 1 0$ </td><td> $5 . 4 0 { \scriptstyle \pm 0 . 7 9 }$ </td></tr><tr><td>Tanh</td><td>683</td><td> $8 8 . 0 6 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 7 8 . 5 5 { \scriptstyle \pm 6 7 . 6 6 }$ </td><td> $8 . 8 5 { \scriptstyle \pm 0 . 6 2 }$ </td></tr><tr><td>IGA</td><td>687</td><td> $9 8 . 5 6 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $7 . 6 4 \pm 1 . 7 0$ </td><td> $0 . 9 8 { \pm } 0 . 1 8$ </td></tr><tr><td>FLAIR</td><td>687</td><td> $9 6 . 2 4 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $8 6 . 7 6 { \scriptstyle \pm 2 8 . 5 0 }$ </td><td> $2 . 8 1 \pm 0 . 3 8$ </td></tr><tr><td>TUNER</td><td>701</td><td> $9 4 . 6 2 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $4 1 . 7 0 { \scriptstyle \pm 1 . 7 5 }$ </td><td> $3 . 3 2 { \pm } 0 . 0 2$ </td></tr><tr><td>VI3NR</td><td>701</td><td> $9 3 . 1 9 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $1 9 0 . 3 8 { \pm } 4 7 . 1 2$ </td><td> $4 . 6 9 { \scriptstyle \pm 0 . 4 7 }$ </td></tr><tr><td>MSA-INR</td><td>697</td><td> $9 8 . 5 9 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 0 . 5 6 { \pm } 1 . 9 3 $ </td><td> $0 . 9 0 { \scriptstyle \pm 0 . 1 1 }$ </td></tr><tr><td>CAFE (enc.)</td><td>709</td><td> $9 3 . 8 1 \pm 1 . 7 9$ </td><td> $5 7 . 8 5 { \scriptstyle \pm 2 6 . 6 0 }$ </td><td> $4 . 5 9 { \pm } 1 . 8 7$ </td></tr><tr><td>CAFE</td><td>715</td><td> $6 3 . 6 2 { \scriptstyle \pm 2 9 . 6 2 }$ </td><td> $1 4 3 0 . 3 0 { \scriptstyle \pm 2 0 6 1 . 9 5 }$ </td><td> $5 6 . 8 8 { \scriptstyle \pm 6 3 . 9 2 }$ </td></tr></table>

Ground truth  
SQMM  
![](images/a4b64452276ca950fa48f48817743bfba3fdb668e71e31cd9daf51efef7a8c1c.jpg)  
Figure 9: The 100 grayscale CIFAR10 images (left) and their SQMM reconstructions (right).