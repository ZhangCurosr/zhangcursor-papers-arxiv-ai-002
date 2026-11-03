# Spacetime Emergence in the Memory of a Learning Agent

Philip A. LeMaitre<sup>1,</sup> <sup>∗</sup> and T. Rick Perche<sup>2,</sup> <sup>†</sup>

<sup>1</sup>University of Innsbruck, Institute for Theoretical Physics, Technikerstrasse 21a, A-6020 Innsbruck, Austria

<sup>2</sup>Nordita, KTH Royal Institute of Technology and Stockholm University, Hannes Alfv´ens v¨ag 12, 23, SE-106 91 Stockholm, Sweden

(Dated: November 3, 2026)

We show that the causal structure of spacetime can emerge within the memory of a learning agent performing a quantum information task. We train a model-based reinforcement-learning agent to maximize the quantum mutual information between localized quantum probes interacting with a quantum field in Minkowski spacetime, without directly providing it with information regarding the spacetime metric or causal structure. In learning to accomplish this task, the reinforcementlearning agent constructs an internal predictive model whose planner develops the characteristic null structure of spacetime, allowing us to reconstruct the causal structure of the Minkowski metric with high accuracy. Within this task, spacetime geometry is obtained from an agent’s successful predictions and actions, providing a first step toward showing that spacetime concepts emerge from operational quantum experience.

## I. INTRODUCTION

General relativity is our most successful description of spacetime and has passed a remarkably broad range of experimental tests to date [1], yet the geometry described by the theory — commonly represented by a Lorentzian metric tensor, which is an object encoding spacetime distances, angles, and causal relationships [2, 3] — is never accessed directly in experiment but is instead inferred from the records left by physical systems interacting with matter and fields [3, 4]. Whether through families of causal trajectories or the responses of localized detectors, every observation provides only finite operational information from which the underlying geometry must be reconstructed [3, 5–11]. This operational viewpoint shifts the emphasis from geometry itself to the physical records encoding that geometry, motivating broader questions that lie at the heart of spacetime reconstruction: what kind of records are suficient for an observer to recover the structure of the underlying spacetime and to what extent do they preserve the geometric information of the spacetime that generated them?

Multiple approaches have been developed to try and answer these questions, with quantum-information-based schemes being of particular interest with regards to what will be considered in the current work (introduced further down). For instance, Ref. [12] reconstructs spatial geometry from the entanglement structure of a Hilbert space factorization by using mutual information to define efective distances before fitting an emergent geometry. Ref. [13] advocates for replacing spacetime distance by correlation strength, treating abstract correlators as the primary structure from which spacetime and fields may arise as approximate descriptions. Additionally, Ref. [14] analyzes how relative distance can be associated with mutual information in entanglement-based emergent-spacetime scenarios while showing that simple entangled systems already contain ambiguities in such reconstructions.

Relativistic quantum information (RQI) arguably provides the most natural framework in which to address this question. In particular, Unruh–DeWitt (UDW) detectors [15, 16] furnish a simple model of localized quantum systems interacting with quantum fields [17, 18] while remaining suficiently tractable that the resulting detector states can be analyzed using the tools of quan tum information theory [19–21]. It is through these interactions that geometric and causal relationships as well as field correlations become encoded into finite dimensional quantum states whose operational properties can be studied directly. UDW systems have consequently become one of the principal tools [22] for investigating how spacetime structure is reflected in quantum information, leading to a variety of protocols for reconstructing geometric quantities from detector data [9, 10, 23].

The perspective adopted here is somewhat diferent to the above mentioned works in that we ask whether a learning agent interacting only with operational quantum data sourced from quantum systems embedded in a spacetime (and with no a priori knowledge of spacetime), can itself acquire an internal representation that reflects the geometry of the spacetime generating those observations. The question is therefore whether successful interaction with a relativistic quantum environment naturally gives rise to representations from which that geometry can subsequently be extracted. Viewed in this way, a learning agent serves as a concrete operational observer whose internal memory is shaped by experience. More concretely, our setting considers model-based reinforcement-learning (RL) agents acting on an environment defined by the outcomes of a quantum information protocol performed in spacetime. RL converts the structure of that environment into a memory of actions that we can attempt to query to ascertain whether it contains a recoverable metric structure.

Our application of machine learning in pursuit of answering this question is a somewhat recent development with only a relatively small number of works employing the same strategy, and fewer still using RL. For example, the authors of Ref. [24] train a model on entanglement features so that the learned tensor-network geometry reproduces the entanglement structure of a quantum many-body state; the authors of Refs. [25, 26] use deep learning in AdS/CFT-inspired settings to infer emergent bulk metrics from field-theoretic or lattice data, and more recently used interpretable neural architectures to reconstruct gravitational metrics from linear-response data [27]; recent holographic reconstruction work uses entanglement-entropy data as input to a machine-learning inverse problem for the bulk metric [28]; and in the RQI sphere, Ref. [29] uses deep learning to distinguish field states, detect field thermalization, and sense spacetime boundaries. Work on quantumenhanced agents and causal machine learning in quantum settings comes closer still by treating agents as active participants whose actions and records mediate what can be learned about a quantum environment [30]. Our approach combines this active participant idea with RL in an RQI setting in a way that answers the original questions, but also distinguishes itself from these other works by drawing attention to the agentic aspect of learning new concepts from acting. In this respect, the problem is also related to work on intuitive physics and physical concept learning, where predictive models acquire internal representations of physical regularities from experience rather than having access to an explicit symbolic theory [31].

To investigate these questions, we construct a detectorgenerated environment in which one Gaussian-smeared UDW detector remains fixed while a second is translated across a two-dimensional grid of spacetime smearing centres $\mathbf { \boldsymbol { x } } ~ = ~ ( \boldsymbol { x } , t )$ Their interaction with a real massless scalar quantum field in (3 + 1)-dimensional Minkowski spacetime, holding two spatial dimensions fixed, generates a transformed quantum mutual-information objective that provides the sole reward to a model-based RL agent with no initial conception of spacetime. The underlying spacetime structure is implicitly encoded in this reward and so the RL agent must learn a predictive representation of the environment — and thus the spacetime structure — through interaction with it. We then ask whether the memory supporting its successful behaviour retains enough of the structure underlying the operational quantum information for the spacetime geometry to be reconstructed from it. For this we apply a mask to the learned planner representation that minimally identifies lightcone-like branches, to which we fit a quadratic form whose coeficients yield the metric tensor components $g _ { t t } \simeq - 0 . 9 9 9 8 8$ and $\dot { g _ { t x } } \simeq 7 . 7 9 \times 1 0 ^ { - 8 }$ (after fixing the conformal freedom by setting $g _ { x x } = 1 )$ with null slopes $d x / d t \simeq \pm 0 . 9 9 9 9 4$ , in close agreement with the corresponding Minkowski values. Since null structure determines the Lorentzian metric only up to a conformal factor [3, 7, 8], this demonstrates that an internal representation learned solely for prediction and action can preserve suficient operational information for the local conformal geometry of the spacetime to then be recovered.

The remainder of the paper is organized as follows. Section II establishes the basics of the UDW model and the framework of Algebraic Quantum Field Theory (AQFT), then derives the relevant quantities for the detectorgenerated reward landscape. Section III briefly introduces the basics of RL and details the architecture of the learning agent. The details of the learning task and RL agent training procedure are then given in Section IV, and subsequently Section V presents the reconstruction of the spacetime metric from the learned representation in the RL agent’s memory. Finally, Section VI discusses the implications, limitations, and possible extensions of the framework. The code used to obtain the results can be found in an accompanying GitHub repository [32].

## II. UNRUH-DEWITT MODEL

The UDW model bridges quantum information with relativity in a concrete way such that a learning task based on quantum information may be phrased in a spacetime setting. From the choice of a spacetime region to couple the detector to and a field state, one can determine the final finite-dimensional detector state whose correlations are then used to construct the reward landscape. The RL agent has access only to this resulting reward, while the spacetime geometry and field-theoretic quantities from which it is generated remain directly inaccessible. Each point in the environment therefore corresponds to a possible spacetime smearing centre for detector 2, with the value assigned to that point determined by the correlations induced through its interaction with the field.

UDW detectors were introduced as probes of quantum fields and observer-dependent particle content [15, 16, 33]. Their appeal in relativistic quantum information comes from the fact that a finite-dimensional quantum system can be coupled locally to a quantum field while keeping the operational output on the level of detector states. This has made the model remarkably useful for studying the transition probabilities, field noise, signalling, covariance, and the dependence of detector response on switching and localization [19–21, 34, 35]; as well as providing a means for operationalizing spacetime properties into quantum information and computation resources for use in entanglement/contextuality harvesting [36–38] and relativistic quantum computing [39–43]. Work also shows how detector models capture fundamental features of the light-matter interaction [44] and how they extend to linearized-gravity settings [45].

For the purpose of this manuscript, we consider two qubit UDW detectors interacting with a real massless scalar quantum field $\hat { \phi } ( { \bf \times } )$ in 3+1-dimensional Minkowski spacetime, whose dynamics are prescribed by the Klein-Gordon equation

$$
\nabla ^ { \mu } \nabla _ { \mu } \hat { \phi } ( \times ) = 0 ,\tag{1}
$$

and can be decomposed as

$$
\hat { \phi } ( { \bf x } ) = \int \frac { \mathrm { d ^ { 3 } } { \cal k } } { \sqrt { 2 ( 2 \pi ^ { 3 } ) | { \cal k } | } } \left( \hat { a } _ { \cal k } ^ { \dagger } e ^ { i ( | { \cal k } | t - { \cal k } \cdot { \bf x } ) } + \hat { a } _ { \cal k } e ^ { - i ( | { \cal k } | t - { \cal k } \cdot { \bf x } ) } \right) ,\tag{2}
$$

where $\hat { a } _ { \pmb { k } } , \hat { a } _ { \pmb { k } } ^ { \dagger }$ are the field ladder operators satisfying the commutation relation $[ \hat { a } _ { \pmb { k } } , \hat { a } _ { { \pmb { k } } ^ { \prime } } ^ { \dagger } ] = \delta ^ { 3 } ( { \pmb { k } } - { \pmb { k } } ^ { \prime } )$

The detectors are modeled as qubits with Hilbert spaces $\mathcal { H } _ { 1 } \cong \mathcal { H } _ { 2 } \cong \mathbb { C } ^ { 2 }$ , each moving along comoving inertial timelike worldlines $z _ { d } ^ { \mu } ( t ) = ( t , \mathbf { \bar { x } } _ { d } ) \mathrm { ~ } , \bar { d } \in \{ 1 , 2 \}$ , where t is an afine inertial time parameter that also represents the detectors’ proper time. Along each worldline, each qubit has internal dynamics governed by a free Hamiltonian $\begin{array} { r } { \hat { H } _ { d } = \frac { \Omega _ { d } } { \Omega } ( \mathbb { 1 } - \hat { \sigma } _ { z } ^ { ( d ) } ) } \end{array}$ , where $\Omega _ { d }$ denotes the energy gap between the ground and excited states $\left| 0 _ { d } \right.$ and $\left| 1 _ { d } \right.$ respectively, so that $\hat { \sigma } _ { z } ^ { ( d ) } = | 0 _ { d } \rangle \langle 0 _ { d } | - | 1 _ { d } \rangle \langle 1 _ { d } |$ . Additionally, each qubit interacts with a massless scalar quantum field $\hat { \phi } ( { \bf \times } )$ locally around their trajectories. In the local description of field theory (see e.g. Ref. [11]), the scalar field $\hat { \phi } ( { \bf \times } )$ satisfies the commutation relation

$$
\left[ \hat { \phi } ( \mathbf { x } ) , \hat { \phi } ( \mathbf { x } ^ { \prime } ) \right] = i E ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathbf { 1 }\tag{3}
$$

where $E ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = G _ { R } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) - G _ { A } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ is called the causal propagator, with $G _ { R } ( \mathsf { x } , \mathsf { x } ^ { \prime } )$ and $G _ { A } ( \mathsf { x } , \mathsf { x } ^ { \prime } )$ being the retarded and advanced propagators of the field, respectively. The causal propagator is related to the Wightman function of the field $W ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \langle \hat { \phi } ( \mathbf { x } ) \hat { \phi } ( \mathbf { x } ^ { \prime } ) \rangle _ { \hat { \rho } _ { \phi } }$ , through the decomposition

$$
\begin{array} { l } { { W ( { \bf x } , { \bf x } ^ { \prime } ) = \displaystyle \frac { 1 } { 2 } \left. \left\{ \hat { \phi } ( { \bf x } ) , \hat { \phi } ( { \bf x } ^ { \prime } ) \right\} \right. _ { \hat { \rho } _ { \phi } } + \displaystyle \frac { 1 } { 2 } \left. \left[ \hat { \phi } ( { \bf x } ) , \hat { \phi } ( { \bf x } ^ { \prime } ) \right] \right. _ { \hat { \rho } _ { \phi } } } } \\ { { { } } } \\ { { { } = \displaystyle \frac { 1 } { 2 } H ( { \bf x } , { \bf x } ^ { \prime } ) + \displaystyle \frac { i } { 2 } E ( { \bf x } , { \bf x } ^ { \prime } ) } } \end{array}\tag{4}
$$

where $H ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \langle \{ \hat { \phi } ( \mathbf { x } ) \hat { \phi } ( \mathbf { x } ^ { \prime } ) \} \rangle _ { \hat { \rho } \phi }$ is called the Hadamard function. Since the field commutator is proportional to the identity operator of the field, it is clear that the causal propagator is independent of the field’s state $\hat { \rho } _ { \phi }$ The causal propagator encodes information that propagates between diferent events through the field. The Hadamard function on the other hand, contains all of the state-dependence (including the quantum degrees of freedom) of the field since diferent field states will not necessarily commute with the anti-commutator of the field.

The qubit-field interaction is prescribed by the interaction Hamiltonian density, in the interaction picture, as

$$
\hat { h } _ { d } ( \mathbf { x } ) = \lambda _ { d } \Lambda _ { d } ( \mathbf { x } ) \hat { \mu } _ { d } ( t ) \otimes \hat { \phi } ( \mathbf { x } ) ,\tag{5}
$$

where the λ is a coupling constant, $\hat { \mu } _ { d } ( t ) = e ^ { i \Omega t } | 1 _ { d } \rangle \langle 0 _ { d } | +$ $e ^ { - i \Omega t } | 0 _ { d } \rangle \langle 1 _ { d } |$ is the time-evolved operator $\hat { \sigma } _ { x } ^ { ( d ) }$ acting on qubit $d ,$ and $\Lambda _ { d } ( \mathsf { x } )$ are spacetime smearing functions, which define the shape of the interaction in spacetime. The functions $\Lambda _ { d } ( \mathsf { x } )$ are each strongly supported around the detectors’ trajectories $z _ { d } ( t )$ . The smearing functions $\Lambda _ { d } ( \times )$ also define smeared versions of the field

$$
\hat { \phi } ( \Lambda _ { d } ) = \int \mathrm { d } ^ { 4 } { \times } \Lambda _ { d } ( { \times } ) \hat { \phi } ( { \times } ) .\tag{6}
$$

The UDW system starts in the initial state $\hat { \rho } ( t _ { 0 } ) { \bf \rho } =$ $\hat { \rho } _ { 1 2 } ( t _ { 0 } ) \otimes \hat { \rho } _ { \phi } = \bigotimes _ { d } ^ { 2 } | 0 \rangle _ { d } \langle 0 | \otimes \hat { \rho } _ { \phi } ,$ where $\hat { \rho } _ { \phi }$ is chosen to be the vacuum state of the field $| 0 \rangle _ { \phi } \langle 0 |$ and $| 0 \rangle _ { d }$ are the ground states of ${ \hat { H } } _ { d } .$ Time evolution of this state to a final time t is given by

$$
\hat { \rho } ( t ) = \hat { U } ( t , t _ { 0 } ) \hat { \rho } ( t _ { 0 } ) \hat { U } ^ { \dag } ( t , t _ { 0 } ) ,\tag{7}
$$

with the time-evolution operators defined by

$$
\hat { U } ( t , t _ { 0 } ) = \mathcal { T } _ { t } \exp \int _ { t _ { 0 } } ^ { t } \mathrm { d } ^ { 4 } \times \hat { h } ( \mathbf { x } ) ,\tag{8}
$$

where $\hat { h } ( \mathbf { x } ) = \hat { h } _ { 1 } ( \mathbf { x } ) + \hat { h } _ { 2 } ( \mathbf { x } )$ is the Hamiltonian density of the full system and $\mathcal { T } _ { t }$ exp denotes the time-ordered exponential with respect to the time parameter t [21].

The spacetime-smearing functions $\Lambda _ { d } ( \mathsf { x } )$ we consider in this work are given by

$$
\Lambda _ { d } ( \mathsf { x } ) = \frac { 1 } { ( 2 \pi \sigma _ { d } ^ { 2 } ) ^ { 3 / 2 } } e ^ { - \frac { ( \mathsf { x } - \bar { \boldsymbol { \pi } } _ { d } ) ^ { 2 } } { 2 \sigma _ { d } ^ { 2 } } - \frac { ( t - \bar { t } _ { d } ) ^ { 2 } } { 2 T _ { d } ^ { 2 } } } ,\tag{9}
$$

where $T _ { d } ~ \in ~ \mathbb { R } _ { + }$ is the temporal width for system $d ,$ $\sigma _ { d } \in \mathbb { R } _ { + }$ is the spatial width for system $d , \ F _ { d } \ \in \ \mathbb { R }$ is the temporal centre for system $d ,$ and $\bar { \pmb { x } } _ { d } \in \mathbb { R } ^ { 3 }$ is the spatial centre vector for system d.

Field-mediated detector correlations have been studied extensively in entanglement harvesting and relativistic quantum communication, where it has been shown that localized detectors can extract correlations from the vacuum and exchange quantum information through a relativistic quantum field [36, 37, 46–49]. This literature separates symmetric and causal field correlations and taskdependent information measures computed from detector states. The environment that the RL agent learns in is an RQI setup of two qubit detectors interacting with a real massless scalar quantum field, with the feedback the

RL agent receives in the form of a reward function that depends on the quantum mutual information:

$$
I ( i : j ) = S ( \hat { \rho } _ { i } ) + S ( \hat { \rho } _ { j } ) - S ( \hat { \rho } _ { i j } ) ,\tag{10}
$$

where $S ( \hat { \rho } _ { i } )$ is the von Neumann entropy for the reduced state of system $i , \hat { \rho } _ { i } \colon$

$$
S ( \hat { \rho } _ { i } ) = - \operatorname { T r } \left( \hat { \rho } _ { i } \log ( \hat { \rho } _ { i } ) \right) .\tag{11}
$$

Notice that both $I ( i : j )$ and $S ( \hat { \rho } _ { i } )$ can be computed explicitly in terms of the spectrum of the operators $\hat { \rho } _ { i } ^ { \phantom { \dagger } }$ $\hat { \rho } _ { j }$ and $\hat { \rho } _ { i j }$

In our protocol, we will consider the case where the detectors have trivial internal dynamics, with $\Omega = 0 .$ . This case is particularly convenient for studying protocols that involve non-infinitesimal coupling constants, as the model can be solved non-perturbatively in these cases [50]. Indeed, in Ref. [51], the final state of two gapless detectors was found in closed form in terms of smeared Hadamard and retarded Green’s functions of the scalar field $\hat { \phi } ( { \bf x } )$ along the detectors’ spacetime smearing functions $\Lambda _ { d } ( \mathsf { x } )$ In particular, if the detector interaction regions are related by a spacetime translation $( \Lambda _ { 1 } ( \mathsf { x } ) = \Lambda _ { 2 } ( \mathsf { x } - \mathsf { v } ) )$ and considering the detectors to be initialized in their ground state $( \hat { \rho } _ { d } ( t _ { 0 } ) = | 0 _ { d } \rangle \langle 0 _ { d } | )$ , their final state at a time t after the interactions $\hat { \rho } _ { 1 2 } ( t ) = \mathrm { T r } _ { \phi } ( \hat { \rho } ( t ) )$ in the $\hat { \sigma } _ { x }$ basis, is given by

$$
\hat { \rho } _ { 1 2 } ( t ) = \frac { 1 } { 4 } \left( \begin{array} { l l l l } { 1 } & { e ^ { - H } e ^ { - 2 \mathrm { i } G _ { 2 1 } } } & { e ^ { - H } e ^ { - 2 \mathrm { i } G _ { 1 2 } } } & { e ^ { - 2 ( H + H _ { 1 2 } ) } } \\ { * } & { 1 } & { e ^ { - 2 ( H - H _ { 1 2 } ) } } & { e ^ { - H } e ^ { 2 \mathrm { i } G _ { 1 2 } } } \\ { * } & { * } & { 1 } & { e ^ { - H } e ^ { 2 \mathrm { i } G _ { 2 1 } } } \\ { * } & { * } & { * } & { 1 } \end{array} \right)\tag{12}
$$

where we defined

$$
\begin{array} { c } { { H = \lambda ^ { 2 } H ( \Lambda _ { 1 } , \Lambda _ { 1 } ) = \lambda ^ { 2 } H ( \Lambda _ { 2 } , \Lambda _ { 2 } ) , } } \\ { { H _ { 1 2 } = \lambda ^ { 2 } H ( \Lambda _ { 1 } , \Lambda _ { 2 } ) = \lambda ^ { 2 } H ( \Lambda _ { 2 } , \Lambda _ { 1 } ) , } } \\ { { G _ { i j } = \lambda ^ { 2 } G _ { R } ( \Lambda _ { i } , \Lambda _ { j } ) , } } \end{array}\tag{13}
$$

with

$$
{ \cal H } ( \Lambda _ { i } , \Lambda _ { j } ) = \int \mathrm { d } ^ { 4 } { \times } \mathrm { d } ^ { 4 } { \times } ^ { \prime } { \cal H } ( { \bf x } , { \bf x } ^ { \prime } ) \Lambda _ { i } ( { \bf x } ) \Lambda _ { j } ( { \bf x } ^ { \prime } ) ,\tag{14}
$$

and similarly for $G _ { R } ( \Lambda _ { i } , \Lambda _ { j } )$

Notice that in Eq. (12), the retarded propagators $G _ { i j }$ yield non-diagonal phases, while the H and $H _ { i j }$ terms are associated with decoherence. Specifically, the H terms determine the local noise experienced by each detector, while the $H _ { i j }$ terms correspond to correlated decoherence terms between them. The corresponding reduced states are

$$
\hat { \rho } _ { 1 } ( t ) = \frac { 1 } { 2 } \left( { { \sigma } _ { e } } ^ { 1 } \cos ( 2 G _ { 1 2 } ) \begin{array} { c } { { e } ^ { - H } \cos ( 2 G _ { 1 2 } ) } \\ { { 1 } } \end{array} \right) ,\tag{15}
$$

$$
\hat { \rho } _ { 2 } ( t ) = \frac { 1 } { 2 } \left( { { \sigma } _ { e } } ^ { 1 } \cos ( 2 G _ { 2 1 } ) \begin{array} { c } { { e } ^ { - H } \cos ( 2 G _ { 2 1 } ) } \\ { { 1 } } \end{array} \right) .\tag{16}
$$

In particular, the eigenvalues of $\hat { \rho } _ { 1 2 }$ are

$$
\begin{array} { c } { { \lambda _ { 1 , 2 } = \displaystyle \frac { 1 } { 4 } - \frac { 1 } { 4 } e ^ { - 2 H } \cosh ( 2 H _ { 1 2 } ) } } \\ { { \mp \displaystyle \frac { 1 } { 4 } e ^ { - 2 H } \sqrt { 4 e ^ { 2 H } \sin ^ { 2 } ( E _ { 1 2 } ) + \sinh ^ { 2 } ( 2 H _ { 1 2 } ) } , } } \end{array}\tag{17}
$$

$$
\begin{array} { l } { { \displaystyle \lambda _ { 3 , 4 } = \frac { 1 } { 4 } + \frac { 1 } { 4 } e ^ { - 2 H } \cosh ( 2 H _ { 1 2 } ) } } \\ { { \displaystyle \qquad \mp \frac { 1 } { 4 } e ^ { - 2 H } \sqrt { 4 e ^ { 2 H } \cos ^ { 2 } ( E _ { 1 2 } ) + \sinh ^ { 2 } ( 2 H _ { 1 2 } ) } , } } \end{array}\tag{8}
$$

with $E _ { i j } = G _ { i j } - G _ { j }$ <sub>i</sub> and the reduced eigenvalues are

$$
\mathrm { S p e c } ( \hat { \rho } _ { 1 } ( t ) ) = \left\{ \frac { 1 } { 2 } \left( 1 \pm e ^ { - H } \cos ( 2 G _ { 1 2 } ) \right) \right\} ,\tag{19}
$$

$$
\mathrm { S p e c } ( \hat { \rho } _ { 2 } ( t ) ) = \left\{ \frac { 1 } { 2 } \left( 1 \pm e ^ { - H } \cos ( 2 G _ { 2 1 } ) \right) \right\} .\tag{20}
$$

We can see that the quantum mutual information only depends on the following parameters: the coupling constant $\lambda ,$ the spatial smearing width $\sigma _ { 1 } = \sigma _ { 2 } = \sigma$ , the temporal smearing width $T _ { 1 } = T _ { 2 } = T$ , the spatial smearing centre $\bar { \mathbf { x } } _ { 2 }$ , and the temporal smearing centre $\bar { t } _ { 2 }$ . In this work we will fix $\lambda , \sigma , T$ while allowing the relative spacetime position of the detectors (encoded in $\bar { t } _ { 2 }$ and $\bar { \bf x } _ { 2 } )$ to be varied.

The core idea of our paper is to allow an RL agent to move the second detector relative to the first, attempting to maximize the quantum mutual information between the two detectors after their interaction with the field. More precisely, we will describe $\bar { t } _ { 2 }$ and $\scriptstyle { \bar { \mathbf { x } } } _ { 2 }$ as values in a spacetime grid, constituting the RL agents’ environment. The RL agent then explores the environment by changing the spacetime placement of the second detector, receiving a reward function based on how well it is able to improve the quantum mutual information through local changes in $\bar { t } _ { 2 }$ and $\scriptstyle { \bar { \mathbf { x } } } _ { 2 }$ . This task updates the RL agent’s model of the environment and updates its memory at each step. We will discuss the task structure and the RL agent’s model in further detail in the following two sections.

## III. REINFORCEMENT-LEARNING AND AGENT ARCHITECTURE

In this section we will introduce the reinforcementlearning (RL) framework and RL agent architecture needed to formulate the learning task and to analyze the RL agent’s learned representation in the next sections. We first introduce some basics of RL that are required for the present work before describing the model architecture used by the RL agent, which is shown schematically in Fig. 1. Readers already familiar with RL may proceed directly to the description of the model architecture below, and those already versed in machine learning more generally are welcome to proceed directly to the next section without loss of understanding.

As we are interested in an agentic perspective on learning spacetime structure, the problem is naturally formulated in terms of RL. The basic agent–environment interaction loop of RL starts with the agent observing a state and deciding to select an action, then receiving a reward as feedback, and finally updating a memory function via a learning update rule whose purpose is to maximize future rewards [52]. In the present setting the state is a spacetime smearing centre for detector 2, the action is a local displacement on the spacetime grid, and the reward is a function of the quantum mutual information, so the RL formulation keeps the operational character of the task explicit.

The RL task is formalized as a Markov Decision Process (MDP), written as the 4-tuple (S, A, R, P). Here S, A, R are the state, action, and reward spaces, respectively; we will generally denote their elements as $s \in S , a \in \mathcal { A } , r \in$ R. The full transition kernel P specifies the probability of each possible next state and reward, given the current state and the chosen action:

$$
P ( s ^ { \prime } , r \mid s , a ) = \mathbb { P } ( S _ { t + 1 } = s ^ { \prime } , R _ { t + 1 } = r \mid S _ { t } = s , A _ { t } = a ) ,\tag{21}
$$

with $S _ { t } , A _ { t } , R _ { t }$ denoting random variables at time t who can take values from the state, action, and reward spaces, respectively. From these quantities we can also define the state-transition kernel (where we keep the symbol $P ,$ but omit its second argument) as

$$
P ( s ^ { \prime } \mid s , a ) = \sum _ { r \in \mathcal { R } } P ( s ^ { \prime } , r \mid s , a )\tag{22}
$$

and the expected reward function from a reward being delivered when transitioning from state s to $s ^ { \prime } , s  \bar { s ^ { \prime } }$ while taking the action $^ { a , }$ as

$$
\begin{array} { l } { { \displaystyle R ( s , a , s ^ { \prime } ) = \mathbb { E } [ R _ { t } \mid S _ { t } = s , A _ { t } = a , S _ { t + 1 } = s ^ { \prime } ] } } \\ { { \displaystyle = \sum _ { r \in \mathcal { R } } r \frac { P ( s ^ { \prime } , r \mid s , a ) } { P ( s ^ { \prime } \mid s , a ) } , } } \end{array}\tag{23}
$$

where $\mathbb { E } [ \cdot | \cdot ]$ is the conditional expectation value. The behaviour of the RL agent (i.e. how it will act given some state) at some time t can be described by the stochastic policy $\pi ( a \mid s ) = \mathbb { P } _ { \pi } ( A _ { t } = a \mid S _ { t } = s )$ , and the RL agent’s goal is typically to maximize their expected discounted return (called the value function)

$$
v _ { \pi } ( s ) = \mathbb { E } _ { \pi } [ G _ { t } \mid S _ { t } = s ]\tag{24}
$$

where

$$
G _ { t } = \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } R _ { t + k + 1 }\tag{25}
$$

is called the discounted return and $\gamma \in [ 0 , 1 )$ is called a discount factor, which controls the strength of future rewards [52, 53]; γ both makes the return finite under standard bounded-reward assumptions and determines how strongly later detector placements influence present actions.

There are two broad ways in which an RL agent can use experience to improve its decisions: model-free valuebased methods and model-based methods. The modelfree value-based methods, such as Q-learning [54, 55], directly learn the expected long-term return associated with taking an action in a state, which works quite well for small- to moderate-sized problems. For large state spaces, these value functions can be more eficiently represented by artificial neural networks, which provide flexible parameterized function approximators capable of generalizing between states rather than storing a separate value for every state-action pair [56–59]. Modelbased methods on the other hand, have the RL agent instead learn an internal predictive model of the environment and use that model to evaluate possible actions before executing them [52].

We adopt the model-based approach to model the RL agent’s memory and decision-making faculties, which in our case, consists of an encoder, decoder, transition model, reward model, and a planning rule that combines their predictions into a decision (see Fig. 1 for how these components fit together). We will refer to the collection of these learned internal models as the agent’s memory, since it is the orchestration of these learned models that retains the necessary behavioural information to act in the environment. This model-based approach is important for our purposes because the learned agent model provides explicit internal objects that can be interrogated after training, meaning that we can also examine whether the representations and predictions supporting its behaviour retain information about the spacetime structure that generated the reward landscape.

To distinguish the physical state from its representations within the agent, let $s _ { t } ~ \in ~ S$ denote the physical environment state and $o _ { t } = o ( s _ { t } ) \in \mathcal { O }$ the normalized observation supplied to the encoder neural network (this will be shown in the next section) at step t. In the present problem, $s _ { t }$ specifies the spacetime smearing centre of detector 2 and $o _ { t }$ is a normalized version of it. The decisionmaking process begins with the encoder, which maps this observation to an internal latent state

$$
z _ { t } = E _ { \theta _ { E } } ( o _ { t } ) \in \mathcal { Z } , \qquad \mathcal { Z } \subseteq \mathbb { R } ^ { d _ { z } } ,\tag{26}
$$

where $\theta _ { E }$ denotes the trainable parameters of the encoder, $d _ { z }$ is the latent dimension, and the latent space $\mathcal { Z }$ is internal to the agent, having elements that are internal vector representations of the observations whose individual components are not assigned any intrinsic physical or geometric meaning. It is the encoder that determines which information about the observation is retained in the internal representation and must preserve enough information to support later dynamics and reward prediction, as well as reconstruction by the decoder.

![](images/54766137ba36eb0d359065aa25aa03f7377f3a87703cafc597dd82c667e9a529.jpg)  
FIG. 1. A schematic of the agent architecture and learning pipeline. Detector 1 is fixed at the origin and the $( x , t )$ grid represents the spacetime smearing centres of detector 2. The thumbnail is Fig. 2, showing the transformed quantum mutual information contained in the reward (Eq. (43)) received by the agent. All eight local actions remain available during planning and exploration, with proposed of-grid moves being clipped by the environment and given a boundary penalty. Here $e _ { a _ { t } }$ is the one-hot representation of the action, and numbers beside network symbols specify hidden-layer widths, while dashed connections show step-replay learning with shared network parameters. The RL agent chooses these actions by initially using the encoder to map the normalized observation $o _ { t }$ to the latent state $z _ { t } = E _ { \theta _ { E } } \big ( o _ { t } \big ) \in \mathbb { R } ^ { 2 0 }$ , then the residual dynamics predicts $\widehat { z } _ { t + 1 } = z _ { t } + F _ { \theta _ { F } } ( z _ { t } , e _ { a _ { t } } )$ which is then fed into the reward model that predicts $\widehat { r } _ { t + 1 } = R _ { \theta _ { R } } \big ( z _ { t } , \widehat { z } _ { t + 1 } , e _ { a _ { t } } \big )$ . Since the planning horizon is $H = 1 .$ the planner score is simply $J _ { \theta } ( s _ { t } , a _ { t } ) = \widehat { r } _ { t + 1 }$ . Reward training uses $z _ { t + 1 } = E _ { \theta _ { E } } ( o _ { t + 1 } )$ , the encoded observed next state, whereas planning uses the model-predicted state $\widehat { z } _ { t + 1 }$ . The decoder acts separately on encoded and predicted latent states. Its two displayed outputs are the reconstructed observation $\widetilde { \sigma } _ { t } ~ = ~ D _ { \theta _ { D } } ( z _ { t } )$ and predicted next observation $\widehat { o } _ { t + 1 } = D _ { \theta _ { D } } ( \widehat { z } _ { t + 1 } )$ ; coordinate reconstruction is also applied to the encoded next observation $z _ { t + 1 }$ . The seven loss constraints $\mathrm { E q s . } \quad ( 4 7 ) , ( 4 9 )$ , and (52)- (55) are shown alongside their corresponding predictions, with sg denoting stop-gradient and $\widehat { z } _ { t } ^ {  }$ the reverse-action prediction obtained from $\widehat { z } _ { t + 1 }$ . The $\lambda _ { j }$ denote the corresponding loss weights. The purple branch first averages the planner action-score fields from the five trained agents into $\overline { { J } } ( s , a )$ and then forms the scalar softmax-average field S. A cell enters the candidate lightcone mask when its score exceeds that of at least 70% of the eight neighbouring cells in a radius-1 square. Cells without a complete neighbourhood are rejected. Least squares then fits the quadratic form to the extracted branch points, with displacements measured relative to the fixed detector and the conformal freedom fixed by $g _ { x x } = 1$ , yielding a representative of the conformal metric in $( t , x )$ coordinate order.

The RL agent’s decision-making then proceeds through planning over this latent representation. A general finitehorizon model-based planner considers candidate action sequences $a _ { 0 : H - 1 }$ comprised of H simulated future actions (H is called the planning horizon), by internally predicting the transitions and rewards generated by each sequence, and subsequently executes the first action of the sequence with the largest predicted discounted return. Here and below, the index t labels actual interactions with the environment, whereas $k = 0 , \ldots , H$ labels model-predicted steps within one of these planning rollouts (defined further down), and we use hats to denote quantities predicted by the learned model. Starting from the encoded current state $\widehat { z } _ { 0 } = z _ { t }$ , the planner generates the model-predicted latent trajectory

$$
\widehat { z } _ { k + 1 } = \widehat { z } _ { k } + F _ { \theta _ { F } } ( \widehat { z } _ { k } , e _ { a _ { k } } ) ,\tag{27}
$$

where $F _ { \theta _ { F } }$ is the learned latent dynamics model with trainable parameters $\theta _ { F }$ and the vector $e _ { a _ { k } }$ is a one-hot encoding of action $a _ { k }$ (meaning that its component corresponding to $a _ { k }$ is one and all remaining components are zero). We choose the form of Eq. (27) because it makes $F _ { \theta _ { F } }$ learn action-dependent displacements in latent space rather than just predicting the next latent state.

For the initial planning step $\widehat { o } _ { 0 } = o _ { t }$ , while subsequent predicted observations are obtained from the decoder $D _ { \theta _ { D } } : \mathcal { Z } \to \mathcal { O }$ , which maps latent states back to observed normalized coordinates and has trainable parameters $\theta _ { D }$ . To distinguish reconstructions of actual observations from model-predicted future observations, we use tildes to denote the reconstruction of the current observation,

$$
\tilde { o } _ { t } = D _ { \theta _ { D } } ( z _ { t } )\tag{28}
$$

whereas a decoded model prediction during planning is denoted with a hat

$$
\widehat { o } _ { k + 1 } = D _ { \theta _ { D } } ( \widehat { z } _ { k + 1 } ) .\tag{29}
$$

The role of the decoder encourages

$$
D _ { \theta _ { D } } ( E _ { \theta _ { E } } ( o _ { t } ) ) \simeq o _ { t } ,\tag{30}
$$

i.e. it encourages that decoded predicted next states should lie near the corresponding observed next coordinates. The decoder anchors the latent representation to the detector coordinates without imposing a particular geometry upon that representation, which is a key distinction because although the encoder receives detector coordinates as input, the latent variables themselves are not amenable to a geometric interpretation a priori.

The last learned component is the reward model $R _ { \theta _ { R } } .$ which during training predicts the next reward given the current latent state z and the encoded observed next latent state $z _ { k + 1 } = E _ { \theta _ { E } } ( o _ { k + 1 } )$ under action $a _ { k } .$ , as

$$
r _ { k + 1 } ^ { \prime } = R _ { \theta _ { R } } ( z _ { k } , z _ { k + 1 } , e _ { a _ { k } } ) ,\tag{31}
$$

with $\theta _ { R }$ denoting its trainable parameters. The planner instead uses the model-predicted transition $\widehat { z } _ { k } \to \widehat { z } _ { k + 1 }$ under action $a _ { k }$ since the future state $z _ { k + 1 }$ is unavailable to them, which is given by

$$
\widehat { r } _ { k + 1 } = R _ { \theta _ { R } } ( \widehat { z } _ { k } , \widehat { z } _ { k + 1 } , e _ { a _ { k } } ) .\tag{32}
$$

This procedure thus channels all of the previous machin ery into building the representation $\widehat { r }$ that is finally used to assign a candidate action sequence the score

$$
J _ { \theta } \big ( s _ { t } , a _ { 0 : H - 1 } \big ) = \sum _ { k = 0 } ^ { H - 1 } \gamma ^ { k } \widehat { r } _ { k + 1 } , \qquad \theta = ( \theta _ { E } , \theta _ { F } , \theta _ { D } , \theta _ { R } ) ,\tag{33}
$$

and then promptly select the action<sup>1</sup>

$$
a _ { t } ^ { \star } = \biggl [ \underset { a _ { 0 : H - 1 } } { \arg \operatorname* { m a x } } J _ { \theta } \left( s _ { t } , a _ { 0 : H - 1 } \right) \biggr ] _ { 0 } .\tag{34}
$$

Importantly, a model-predicted rollout here refers to a sequence generated internally by the learned dynamics and reward models rather than one actually executed in the environment. In the present work we use the planning horizon $H = 1$ , meaning that the planner predicts the next latent state and immediate reward associated with each valid local action from the current encoded state $z _ { t } ,$ then choosing the action with the largest predicted reward

$$
a _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } R _ { \theta _ { R } } ( z _ { t } , z _ { t } + F _ { \theta _ { F } } ( z _ { t } , e _ { a } ) , e _ { a } ) .\tag{35}
$$

Thus, although the observations supplied to the encoder are detector coordinates, action evaluation is mediated entirely by the learned latent representation and predictive models.

To learn the consequences of a broad range of actions, the RL agent must sometimes explore alternatives to the planner’s recommendations. The agent therefore follows the planner’s recommendation with probability 1 − ϵ and selects an action uniformly at random with probability ϵ, where $0 \leq \epsilon \leq 1$ . For a specified constant ϵ, this is implemented by drawing a number u uniformly from [0, 1]: if $u \geq \epsilon ,$ the agent follows the planner’s recommendation; otherwise, it samples an action uniformly from the available actions. This procedure is known as ϵ-greedy exploration. An exploration schedule is also implemented, in which ϵ varies with the cumulative number n of environment steps according to $\epsilon ( n ) = 0 . 1 + 0 . 9 e ^ { - n / \epsilon _ { \mathrm { d e c a y } } }$ . This schedule starts at $\epsilon = 1$ and gradually approaches 0.1, so that early training is dominated by uniformly random actions that populate the replay memory with a broad range of local transitions (more on this below), while later training increasingly uses the planner’s output.

Figure 1 demonstrates how the learning process unfolds through each of the above components. The training and associated loss functions will be described in the next section.

## IV. LEARNING TASK

In this section we introduce and describe the details of the learning task and parameter choices of the RL agent architecture. The question we are interested in addressing in this work is whether a learning agent performing a quantum information task that is embedded in spacetime with no a priori notion of spacetime — can develop a representation of that spacetime within its memory after successfully completing the task. More concretely, given the predictive models and behaviour policy of a (modelbased) RL agent tasked with maximizing a monotonic function of the quantum mutual information between two UDW qubit systems, both embedded in a spacetime, can we extract the causal or metric structure of the spacetime from them? The simplicity of the UDW model, combined with how it connects quantum information and spacetime geometry, makes it an ideal choice for setting up the problem environment. Note that from now on, we will hold two of the spatial dimensions of the $( 3 + 1 )$ dimensional spacetime fixed and consider a (1 + 1)-dimensional slice.

In our setting, the MDP that represents the task, $( \mathcal { S } , \mathcal { A } , \mathcal { R } , P )$ , has a state space $s$ consisting of a $3 0 1 \times 3 0 1$ grid whose cells index spacetime smearing centres

$$
\begin{array} { r } { \mathbf { \boldsymbol { x } } = ( \boldsymbol { x } , t ) , \qquad \boldsymbol { x } , t \in [ - 1 0 , 1 0 ] , } \end{array}\tag{36}
$$

for detector 2, with detector 1 fixed at the origin. If $i _ { x } , i _ { t } \in \{ 0 , \ldots , 3 0 0 \}$ , the physical coordinates are

$$
x ( i _ { x } ) = - 1 0 + i _ { x } \Delta ,\tag{37}
$$

$$
t ( i _ { t } ) = - 1 0 + i _ { t } \Delta ,\tag{38}
$$

$$
\Delta = { \frac { 2 0 } { 3 0 1 - 1 } } ,\tag{39}
$$

and the observed input to the model at interaction step t is the normalized coordinate representation

$$
o _ { t } = o ( s _ { t } ) = \left( \frac { x + 1 0 } { 2 0 } , \frac { t + 1 0 } { 2 0 } \right) \in [ 0 , 1 ] ^ { 2 } .\tag{40}
$$

The goal state of the task corresponds to the centre of the physical grid $( 0 , 0 ) \in [ - 1 0 , 1 0 ] ^ { 2 }$ where the detectors interaction regions coincide. Since quantum mutual information measures the total information content of the joint system excluding correlations between subsystems, it is maximized in our setting when the detectors’ interactions are as close as possible; the reason being that field-mediated correlations and communication can decay rapidly with detector separation. An episode terminates when the RL agent reaches this goal state, corresponding to the maximum transformed quantum mutual information $\tilde { I } _ { \mathrm { m a x } }$ , or when the maximum number of allowed steps is reached. An episode is considered successful if the RL agent reaches the goal state before this maximum.

The action set $\mathcal { A }$ consists of the eight nearest-neighbour grid moves

$$
\mathcal { A } = \{ ( \delta _ { x } , \delta _ { t } ) : \delta _ { x } , \delta _ { t } \in \{ - 1 , 0 , 1 \} \} \backslash \{ ( 0 , 0 ) \} ,\tag{41}
$$

![](images/7c79f1213d80df3ce93fc0bab31c5c2c60770a9fdeb9d8615c47f64013f5949b.jpg)  
FIG. 2. The transformed quantum mutual information landscape Eq. (42), navigated by the agent. The plot shows the broad central high-information region corresponding to small detector separation and the diagonal ridge structure associated with the light cone that the RL agent samples only through the reward.

so that an action changes the integer cell by $( i _ { x } ~ +$ $\delta _ { x } , i _ { t } + \delta _ { t } )$ , corresponding to a physical displacement of $( \Delta x , \Delta t ) = \left( x ( i _ { x } + \delta _ { x } ) - x ( i _ { x } ) , t ( i _ { t } + \delta _ { t } ) - t ( i _ { t } ) \right)$ . Notice that each state in S corresponds to a run of the experiment where the separation between the detectors is (x, t). The agent’s decision to move the relative interaction centres by $( \Delta x , \Delta t )$ then corresponds to another run of the experiment associated with the new relative separations $( x + \Delta x , t + \Delta t )$ The environment also prevents proposed moves that would leave the grid and assigns a corresponding boundary penalty $b _ { t } = 0 . 2 { \bf 1 } _ { \mathrm { b o u n d a r y } } ( s _ { t } , a _ { t } )$ The spacetime grid transition itself is deterministic, and the stochasticity during training comes from the ϵ-greedy exploration rule rather than from stochastic spacetime dynamics or from sampling the detector state.

To increase the contrast and visibility of the quantum mutual information further away from the centre of the spacetime grid without breaking its monotonicity as an information measure, we use the normalized power transform of the quantum mutual information

$$
\tilde { I } ( { \bf x } ) = \left[ \frac { I ( \times ) } { \operatorname* { m a x } _ { \bf y } I ( \bf y ) } \right] ^ { p } , \qquad p = \frac { 1 } { 4 } .\tag{42}
$$

The detector parameters used for this landscape are<sup>2</sup> $T =$ 0.15, $\sigma = 0 . 1 5$ , and $\lambda = 1$ , with the initially uncorrelated product detector state given in Sec. II. The bare quantum mutual information range is $[ 3 . 1 9 2 9 \times 1 0 ^ { - 8 } , 6 . 5 4 7 \hat { 7 } \times 1 0 ^ { - 2 } ]$ and the transformed range is approximately [0.02643, 1]. Figure 2 visualizes this function, highlighting the dependence of the quantum mutual information on the properties of spacetime through the central, diagonal lightcone ridges.

The reward from the environment is comprised of four terms:

$$
\begin{array} { r } { R ( s _ { t } , a _ { t } , s _ { t + 1 } ) = \left[ \tilde { I } ( s _ { t + 1 } ) - \tilde { I } _ { \mathrm { m a x } } \right] + 0 . 0 2 \mathbf { 1 } _ { \Delta \tilde { I } _ { t } \geq 0 } } \\ { - 0 . 0 5 \mathbf { 1 } _ { \Delta \tilde { I } _ { t } < 0 } - b _ { t } , \qquad } \end{array}\tag{43}
$$

where the realized reward is $r _ { t + 1 } = R ( s _ { t } , a _ { t } , s _ { t + 1 } )$ ， $b _ { t }$ is the boundary penalty defined previously, $\Delta \tilde { I } _ { t } = \tilde { I } ( s _ { t + 1 } ) -$ $\tilde { I } ( s _ { t } ) , \tilde { I } _ { \mathrm { m a x } } = 1$ in this case, and no additional terminal reward is assigned upon reaching the goal state. The first term merely rewards the RL agent for being in large magnitude states and subtracting of the maximum from it essentially creates a penalty field, while termination upon reaching the maximum prevents the RL agent from exploiting the reward by hovering near the goal state. The next two terms, $0 . 0 2 { \bf 1 } _ { \Delta \tilde { I } _ { t } \geq 0 }$ and $- 0 . 0 5 { \bf 1 } _ { \Delta \tilde { I } _ { t } < 0 }$ , measure the progress of the RL agent towards the goal state, with the former term giving a small positive reward for moves that do not decrease the quantum mutual information and the latter term giving a slightly larger penalty for moves that do. The reward inherits the dependence on spacetime structure from the quantum mutual information, and since the RL agent does not know what this function is or represents, from the agent’s perspective, this feedback is only implicitly associated with said structure.

The purpose of using a model-based architecture in this work is to add an extra element of genuine discovery of the environment structure by the agent. The reward already masks the explicit structure, but the RL agent still receives spacetime coordinates, and so the job of the model is to first translate them into an internal latent state via the trainable encoder (only implicitly dependent on spacetime coordinates), whereas the job of the decoder is to make this representation somewhat faithful. After this point, the RL agent does all of its decisionmaking based on the internal latent states: training a reward model of the environment reward and a latent dynamics model of the environment transition probability to simulate the environment from their own internal viewpoint. In this sense, the model really can be argued to be discovering the spacetime structure autonomously from scratch.

During training, the RL agent learns to navigate the environment by updating its models from feedback via the minimization of a composite loss function over uniformly sampled experience replay mini-batches. Experience replay works as a memory bufer that holds observed transitions $\left( { { s _ { t } } , { a _ { t } } , { r _ { t + 1 } } , { s _ { t + 1 } } } \right)$ —-including episodeboundary markers so that sampled windows do not cross resets—-from which randomly sampled subsets of onestep transitions or mini-batches are subsequently used for parameter updates [59, 60]. A sampled mini-batch contains transitions $\left( o _ { b , k } , a _ { b , k } , r _ { b , k + 1 } , o _ { b , k + 1 } \right)$ consisting of the current normalized coordinates $^ { O } b , \boldsymbol { k } \cdot$ , action $^ { a _ { b , k } , }$ realized reward ${ r } _ { b , k + 1 } .$ and next normalized coordinates $^ { O _ { b , k + 1 } }$ , with batch index $b .$ We denote encoded latent states $z _ { b , k }$ and predicted next latent states $\widehat { z } _ { b , k + 1 }$ by

$$
z _ { b , k } = E _ { \theta _ { E } } ( o _ { b , k } ) ,\tag{44}
$$

$$
z _ { b , k + 1 } = E _ { \theta _ { E } } ( o _ { b , k + 1 } ) ,\tag{45}
$$

$$
\widehat { z } _ { b , k + 1 } = z _ { b , k } + F _ { \theta _ { F } } \left( z _ { b , k } , e _ { a _ { b , k } } \right) ,\tag{46}
$$

and use $\langle \cdot \rangle$ for the uniform average over the sampled replay mini-batch. The norm we will use for the loss functions below is the mean-square norm $\begin{array} { r } { \| \boldsymbol { v } \| ^ { 2 } = d ^ { - 1 } \sum _ { i = 1 } ^ { d } v _ { i } ^ { 2 } } \end{array}$ for vectors $\boldsymbol { v } \in \mathbb { R } ^ { d }$ . The composite loss function is a linear combination of a series of individual loss functions that play diferent roles. Let us now go through each term of the composite loss function.

The first loss term is the reward-prediction loss, and it trains the local reward model $r _ { b , k + 1 } ^ { \prime }$ on the immediate realized reward $r _ { b , k + 1 } \mathrm { : }$

$$
\mathcal { L } _ { r } = \left. \left( r _ { b , k + 1 } ^ { \prime } - r _ { b , k + 1 } \right) ^ { 2 } \right. ,\tag{47}
$$

where

$$
\boldsymbol { r } _ { b , k + 1 } ^ { \prime } = R _ { \theta _ { R } } \big ( z _ { b , k } , z _ { b , k + 1 } , e _ { a _ { b , k } } \big )\tag{48}
$$

is computed from $z _ { b , k } , \ z _ { b , k + 1 }$ , and $^ { a _ { b , k } }$ . The next loss term is the latent dynamics loss, and it trains the latent transition model to land near the encoded next observation:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d y n } } = \left. \left| \left| \widehat { z } _ { b , k + 1 } - \mathrm { s g } ( z _ { b , k + 1 } ) \right| \right| ^ { 2 } \right. , } \end{array}\tag{49}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operation, which blocks the flow of gradients during optimization so that the encoded target is detached in the main dynamics term and the transition model does not chase a moving encoder. The next two loss terms act as consistency constraints, keeping this separation from becoming a loophole. If $a _ { b , k } ^ { - 1 }$ denotes the reverse grid action, then the corresponding latent dynamics update is

$$
\begin{array} { r } { \widetilde { z } _ { b , k } ^ { - } = \widehat { z } _ { b , k + 1 } + F _ { \theta _ { F } } \left( \widehat { z } _ { b , k + 1 } , e _ { a _ { b , k } ^ { - 1 } } \right) , } \end{array}\tag{50}
$$

and the loss terms are given by

$$
\mathcal { L } _ { \mathrm { d y n , c y c } } =  | | \widetilde { z } _ { b , k } ^ {  } - z _ { b , k } | | ^ { 2 }  ,\tag{51}
$$

$$
\mathcal { L } _ { \mathrm { e n c , d y n } } = \left. \left| \left| \mathrm { s g } ( \widehat { z } _ { b , k + 1 } ) - z _ { b , k + 1 } \right| \right| ^ { 2 } \right. .\tag{52}
$$

The first term enforces an imagined forward step followed by the reverse action to return to the original latent state, while the second term applies the same transition mismatch with the predicted latent state detached, so that the encoder is also pressured to remain compatible with the latent dynamics model. The last batch of loss terms are associated with the decoder, and their job is to anchor the latent state to the observed coordinates:

![](images/c85eddded4c56145a8e2743fbade7c49e23d714f0e74f30012583136b562cedd.jpg)

![](images/ff25fecad6df21a2115ae98d6eaa5e4fa5bfcf2e81ba85eec9715e16ec116532.jpg)  
FIG. 3. Learning curves for the model. On top is the episode return and below that is the episode length over 4000 episodes, both showing rapid convergence to the optimal policy and stability thereafter. Each plot is an average over 5 agents with the shaded regions representing the standard deviation around the average curve with respect to the 5 agents.

$$
\mathcal { L } _ { \mathrm { d e c } } = \left. \left. D _ { \theta _ { D } } ( z _ { b , k } ) - o _ { b , k } \right. ^ { 2 } \right. ,\tag{53}
$$

$$
\mathcal { L } _ { \mathrm { d e c , d y n } } = \left. \left. D _ { \theta _ { D } } ( \widehat { z } _ { b , k + 1 } ) - o _ { b , k + 1 } \right. ^ { 2 } \right. ,\tag{54}
$$

$$
\mathcal { L } _ { \mathrm { c y c } } = \left. \Vert E _ { \theta _ { E } } ( D _ { \theta _ { D } } ( z _ { b , k } ) ) - z _ { b , k } \Vert ^ { 2 } \right. .\tag{55}
$$

The first term reconstructs the normalized coordinate features from encoded latent states, the second term tries to constrain predicted next latent states to decode near the observed next coordinate, and the third term prevents the decoder from being accurate only after leaving the encoder’s latent representation. Putting all these loss terms together, the composite loss function is therefore

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { r } + 2 . 5 \mathcal { L } _ { \mathrm { d y n } } } \\ & { \phantom { \mathcal { L } _ { \mathrm { t o t a l } } } + 0 . 2 \mathcal { L } _ { \mathrm { d e c } } + 0 . 2 \mathcal { L } _ { \mathrm { d e c , d y n } } + 0 . 0 5 \mathcal { L } _ { \mathrm { c y c } } } \\ & { \phantom { \mathcal { L } _ { \mathrm { t o t a l } } } + 0 . 5 \mathcal { L } _ { \mathrm { d y n , c y c } } + 0 . 8 \mathcal { L } _ { \mathrm { e n c , d y n } } . } \end{array}\tag{56}
$$

As is standard practice when working with deep neural networks, the weights attached to the loss function terms are tuned so the learning scheme can be successful. The largest weight falls on the dynamics term because the planner needs a usable one-step latent model before its score surface can carry meaningful action information. The reward loss ties the latent transition to the environ ment reward, which is crucial for properly navigating the environment and completing the task, so it has a weight of 1. The latent transition loss terms also have relatively large weights because of the previously stated importance of the transition model. Finally, the decoder loss terms keep the latent representation tied to the observed detector grid, which is useful but not as important as the other terms, so they have small weights.

The training hyperparameters we used are as follows. The run uses $N _ { \mathrm { e p } } = 4 0 0 0$ episodes with at most $T _ { \mathrm { m a x } } =$ 4000 environment steps per episode, which fixes the maximum number of detector placements the RL agent can sample before an episode terminates. Since the planner uses horizon $H ~ = ~ 1$ , its score contains only the predicted immediate reward, so $\gamma = 0 . 9 7$ does not affect action selection and the reported episode return is likewise the undiscounted sum $\bar { \sum _ { t } } r _ { t + 1 }$ of the shaped rewards. As we mentioned earlier, exploration is ϵ-greedy with $\epsilon _ { \mathrm { s t a r t } } ~ = ~ \epsilon ( 0 ) ~ = ~ 1 . 0 , ~ \epsilon _ { \mathrm { f i n a l } } ~ \sim ~ 0 . 1$ (with a maximum number of steps $n _ { \mathrm { m a x } } ~ = ~ N _ { \mathrm { e p } } T _ { \mathrm { m a x } } )$ , and a stepbased decay constant $\epsilon _ { \mathrm { d e c a y } } = 8 . 0 \times 1 0 ^ { 4 }$ . Episode starts are sampled uniformly during the first 35% of training, from a 70% uniform and 30% edge-biased mixture during the next 35%, and from a 40% uniform and 60% edgebiased mixture during the final phase. In each phase, the stated probabilities first select the corresponding region and the starting state is then sampled uniformly from that region, with edge starts drawn from a band of 25 cells near the grid boundary. This start schedule forces the planner to learn from longer moves across the finite grid rather than only from starts near the central high-information region, which assists in grid cov erage and reinforcing far away states. The latent di mension is set to $d _ { z } = 2 0$ so the internal latent representation can be rich enough. The stochastic gradient descent optimization is implemented by the AdamW algorithm from PyTorch with a batch size for experience replay of $B = 2 5 6$ , a learning rate for the AdamW step size of $\eta = 1 0 ^ { - 4 }$ , a weight decay of $\lambda _ { \mathrm { w d } } = 1 0 ^ { - 5 }$ , and a gradient-norm cutof of $\| \nabla \| _ { \operatorname* { m a x } } = 7$ to prevent occasional high-error replay batches from destabilizing the learned model. The stated weight decay is applied to multidimen sional weights, while biases and one-dimensional parameters have zero weight decay, otherwise all other AdamW settings retain their PyTorch defaults. The step-replay memory has capacity $\dot { C } _ { \mathrm { r e p l a y } } = 5 \times 1 0 ^ { 4 }$ , and replay updates begin once it contains $N _ { \mathrm { r e p l a y } } ^ { \mathrm { m i n } } = 4 0 0 0$ usable transition windows. Each eligible environment step performs one gradient update, and after the first 50 episodes, the update interval is selected adaptively from (1, 4, 8, 16, 32) steps using the preceding 50 episodes. The adaptive routine requires a recent success rate of at least 0.9, a recent mean episode length no more than 1.25 times the lower quartile among successful episodes, a recent mean return no more than 0.10 robust return-scale below the historical upper quartile, and a recent mean prediction error no more than 1.25 times its historical median. The interval is set by the number of satisfied criteria, with a failure returning the schedule to the initial one-step interval. Each of the predictive objects uses a multi-layered neural network with SiLU activation functions

$$
\mathrm { S i L U } ( x ) = { \frac { x } { 1 + e ^ { - x } } }\tag{57}
$$

and linear output layers. The encoder maps the twodimensional observation to the $d _ { z } = 2 0$ dimensional latent state, while the decoder maps the latent state back to the two-dimensional observation. The encoder, decoder, and reward models all have hidden-layer widths of (64, 64), and the latent dynamics model uses (128, 128). From the definitions above, the dynamics model takes the latent state and one-hot action as input and outputs a $d _ { z }$ -dimensional latent displacement, while the reward model takes the current and encoded observed next latent states together with the one-hot action and outputs a single predicted reward.

Training is performed for five diferent agent-parameter and environment random-generator seed initializations, and the final diagnostic fields used for plotting and metric recovery are averaged over these trained runs. The RL agent seed controls the network initialization, ex ploration, replay mini-batch sampling, and planner sampling, and edge/arm curriculum starts, while the environment seed controls selection of the starting state. Af ter each training run, the network parameters are frozen and the model is evaluated over 1000 episodes with exploration disabled and starting states sampled uniformly from the full grid with a separate seed. The maximum number of allowed steps remains $T _ { \mathrm { m a x } } = 4 0 0 0$ , and an evaluation episode is successful when the RL agent reaches the goal state (0, 0) before this limit. Every trained RL agent reaches a success rate of 1.0 and a mean final objective $\tilde { I } = 1 . 0$ . The average of the five evaluation summaries gives a mean episode length of approximately 103.85 and a mean undiscounted shaped return of approximat $\mathrm { e l y - } 7 6 . 8 5$ The training loop can be visualized in Fig. 1 and the learning curves in Fig. 3.

## V. GEOMETRIC FEATURES OF THE AGENT MEMORY

Figure 3 shows that the RL agent has essentially converged to the optimal policy for completing the task. If the predictive models supporting this converged policy have retained the spacetime dependence encoded indirectly in the reward landscape, this structure should therefore be detectable in its transition $F _ { \theta _ { F } }$ and reward $R _ { \theta _ { R } }$ model memories as well as its internal latent state. The planner uses these learned objects to predict the outcomes of candidate local actions and optimize the score $\operatorname { E q } .$ (33), allowing us to explicitly inspect whether its decisions retain information about the causal structure underlying the detector environment. The question is then: do the relations within these objects retain enough information for the causal geometry underlying the detector environment to be reconstructed? We train five independent agents, indexed by $m = 1 , \ldots , 5$ , and average the corresponding diagnostic fields over these agents unless otherwise stated.

We will study the memory of the planner to try and answer the above question using two fields that provide complementary views of this learned memory. For each spacetime grid point $\mathbf { \boldsymbol { x } } = ( \boldsymbol { x } , t )$ , let $s ( \times )$ denote its corresponding environment state. The first diagnostic is the best-action planner score

$$
\overline { { B } } ( \mathsf { x } ) = \frac { 1 } { 5 } \sum _ { m = 1 } ^ { 5 } B ^ { ( m ) } ( \mathsf { x } ) ,\tag{58}
$$

which is the average over the best-action planner scores (Eq. (33)) for each RL agent m:

$$
B ^ { ( m ) } ( \times ) = \operatorname* { m a x } _ { a \in \mathcal { A } } J _ { \theta ^ { ( m ) } } ( s ( \times ) , a ) ,\tag{59}
$$

where the active settings use the full eight-action set A at every x, so this assigns to each detector placement the score of its most favourable action $a _ { t } ^ { \ast } ~ ( \mathrm { E q . } ~ ( 3 4 ) )$ . The second diagnostic is the expected action obtained from a softmax distribution over planner scores:

$$
\bar { \xi } _ { \theta ^ { ( m ) } } ( \mathsf { x } ) = \sum _ { a \in \mathcal { A } } \xi _ { a } \frac { \exp [ \beta J _ { \theta ^ { ( m ) } } \left( s ( \mathsf { x } ) , a \right) ] } { \displaystyle \sum _ { b \in \mathcal { A } } \exp [ \beta J _ { \theta ^ { ( m ) } } \left( s ( \mathsf { x } ) , b \right) ] } ,\tag{60}
$$

where ${ \xi } _ { a } = ( \delta _ { x } , \delta _ { t } )$ is the grid displacement associated with action a and $\beta$ acts like an inverse temperature parameter that governs how spread out the distribution is. We average this field over the five agents and use it to visualize how the relative action scores determine an overall preferred direction of motion. For the geometric reconstruction, we first average the action scores themselves

$$
\overline { { J } } ( \mathsf { x } , a ) = \frac { 1 } { 5 } \sum _ { m = 1 } ^ { 5 } J _ { \theta ^ { ( m ) } } ( s ( \mathsf { x } ) , a ) ,\tag{61}
$$

and then form the scalar softmax-average score

$$
\overline { { S } } ( \mathsf { x } ) = \sum _ { a \in \mathcal { A } } \overline { { J } } ( \mathsf { x } , a ) \frac { \exp [ \beta ^ { \prime } \overline { { J } } ( \mathsf { x } , a ) ] } { \displaystyle \sum _ { b \in \mathcal { A } } \exp [ \beta ^ { \prime } \overline { { J } } ( \mathsf { x } , b ) ] } ,\tag{62}
$$

with $\beta ^ { \prime } = 2$ giving the best results for the reconstruction procedure.

Figure 4 shows these fields after the training. The softmax expected-action field illustrates how the planner directs the RL agent towards higher-information regions along gradient-like flows according to the underlying spacetime structure present in Fig. 2, which is most visible amongst the trajectories moving along or into the diagonal arms of the lightcone. The best-score field also displays these features and closely resembles Fig. 2 in structure, preserving both its broad central support and distinctive diagonal ridges. The best-score field additionally highlights that planning does alter the original landscape, most noticeably through thicker diagonal arms and widening diagonal edges. A likely explanation for the first is that wider favourable regions can receive high scores whenever one of the available actions leads towards them, while the second may arise from propagation of a weaker signal that makes it harder for the RL agent to diferentiate the optimal path. However, the two plots put together provide us with an approximate representation of the spacetime structure.

We can now inspect the quality of the internal model acquired by the RL agent. There are a few useful measures we can use to assess this quality, which we plot in Fig. 5. The first two plots (a)—(b) show the two largest encoder principal components obtained from the encoded detector-grid observations, which indicate the two directions accounting for the greatest variation in the latent representation. Both have a clear boundary that they vary smoothly across over the detector grid, indicating that the latent representation retains spatial variation relevant to planning. The next plot (c) shows the mean one-step latent dynamics error between the predicted next latent state $\widehat { z } _ { t + 1 }$ and encoded observed next state $z _ { t + 1 }$ , averaged over the available actions at each grid point, which we see remains essentially negligible across the explored region. The decoder reconstruction error in plot (d) is given by

$$
\varepsilon _ { \mathrm { d e c } } ( \mathsf { x } ) = \| D _ { \theta _ { D } } \bigl ( E _ { \theta _ { E } } ( o ( \mathsf { x } ) ) \bigr ) - o ( \mathsf { x } ) \| ,\tag{63}
$$

and shows that the latent states remain anchored to the observed detector-placement grid in the regions of interest, also demonstrating the poor edge representation and pointing to a root cause of bad transition model predictivity on edges. Plot (e) displays the planner action gap

$$
\Delta J _ { \theta } ( \mathsf { x } ) = J _ { \theta } ^ { ( 1 ) } ( \mathsf { x } ) - J _ { \theta } ^ { ( 2 ) } ( \mathsf { x } ) ,\tag{64}
$$

where $J _ { \theta } ^ { ( 1 ) } ( \times ) \geq J _ { \theta } ^ { ( 2 ) } ( \times )$ are the two largest action scores at that grid point. This indicates that the best action to take along the lightcone branches is distinct relative to the next best, corroborating the story from Fig. 4. The planner selected-action reward in the final plot (f) exhibits the same lightcone-like support learned from the detector environment. Together, the plots in Fig. 5 show that the RL agent has developed a predictively useful internal model whose latent representation, dynamics, and action preferences retain structured information about the detector environment. Further diagnostics are collected in Appendix A.

Figures 4 and 5 clearly show the remnants of the spacetime structure exhibited in Fig. 2. We will now use the local structure of $\overline { S }$ to recover the causal geometry of the underlying spacetime, which in this case is flat (1 + 1)-dimensional Minkowski spacetime with metric tensor components $g _ { t t } = - 1 , g _ { t x } = g _ { x t } = 0 , g _ { x x } = 1$ and eigenvalues $v _ { \pm } ~ = ~ \pm 1$ To do this, we will define a procedure to extract the relevant lightcone data $\{ ( \Delta t _ { i } , \Delta x _ { i } ) \}$ and then use it to determine the coeficients of the quadratic form

$$
q ( \Delta t , \Delta x ) = g _ { t t } \Delta t ^ { 2 } + 2 g _ { t x } \Delta t \Delta x + g _ { x x } \Delta x ^ { 2 } ,\tag{65}
$$

by solving for the null (lightlike) case $q ( \Delta t , \Delta x ) = 0$

In (1 + 1) dimensions, two null directions determine the conformal class of a Lorentzian metric, and so writing $v = d x / d t$ , the null condition becomes

$$
g _ { t t } + 2 g _ { t x } v + g _ { x x } v ^ { 2 } = 0 .\tag{66}
$$

For lightcone branch slopes $v _ { + }$ and $v _ { - }$ , their sum and product satisfy

$$
v _ { + } + v _ { - } = - \frac { 2 g _ { t x } } { g _ { x x } } ,\tag{67}
$$

![](images/e3968cb89cd4a997be97e690b54b747009cf24143bac7a345fa542d6dcd9cdae.jpg)

![](images/cc4878cc22ed9a50a9bea526d5566e1c9e1f1a08943b77821ca890e63278e11c.jpg)  
FIG. 4. Average planner memory output plots. Left: softmax expected-action field using $\beta = 0 . 7$ and normalized arrows, with the choice of $\beta$ used only for visualization and not entering into training or planning. Right: averaged best planner score B at each detector placement. The trained planner maps the detector-generated reward landscape to a local action flow and a score surface that retain the underlying spacetime features (especially visible are the diagonal lightcone arms) inherited from the field correlations appearing in the quantum mutual information Eq. (10).

![](images/73d4977335e0b485d3c346d52cb57eae50a1f7d2086a956f101e4a3f892ed068.jpg)

![](images/6d631372d885723a53106e4ee779cc1400b8dd0ad8acb061dbcd6978ec1d1d9d.jpg)

![](images/5187d7f6cc676652366990db3e07abc6cbc5a6754014f777c66d72cb12af5212.jpg)

![](images/d70364752fc172eb4d9914265f21ae0df1b5158845ed7d90d1c2057bf5d1ea22.jpg)

![](images/2df5ea5a46bb9fc9a56461bd6ba1ccf4ce25830d434852044a7668212ef8ac5f.jpg)

![](images/3ce7407c5b09fb4e19faef289435eb6d8a3421ce7ae3727610608d9f3050a993.jpg)  
FIG. 5. Internal diagnostics for the trained planner. Panels (a) and (b) show the first two principal components of the latent encoder; panel (c) shows the mean one-step latent dynamics error; panel (d) shows the current-state decoder reconstruction error; panel (e) shows the planner action gap; and panel (f) shows the planner-step reward on the selected action. Together, these quantities describe the latent representation, its one-step predictive dynamics, its connection to the observed detectorplacement space, and the action preferences produced by the planner.

$$
{ v } _ { + } { v } _ { - } = \frac { g _ { t t } } { g _ { x x } } ,\tag{68}
$$

with conformal freedom in the choice of $g _ { x x }$ that we

choose to be $g _ { x x } = 1$ , after which the lightcone branch geometry determines $g _ { t t }$ and $g _ { t x }$ . For the extracted lightcone branch points $( \Delta t _ { i } , \Delta x _ { i } )$ , these coeficients are ob-

tained with equal weighting from the least-squares problem

$$
\left( \widehat { g } _ { t t } , \widehat { g } _ { t x } \right) = \arg \operatorname* { m i n } _ { g _ { t t } , g _ { t x } } \sum _ { i } \left( g _ { t t } \Delta t _ { i } ^ { 2 } + 2 g _ { t x } \Delta t _ { i } \Delta x _ { i } + \Delta x _ { i } ^ { 2 } \right) ^ { 2 } .\tag{69}
$$

We now need to find a good way of defining what points lie on the lightcone branches, as seen from the agent’s memory. Since the planner assigns value according to locally accessible outcomes, the central high-reward region extends into neighbouring cells from which favourable states can be reached, so an absolute score threshold then selects much of this broad basin. The diagonal lightcone branches are more clearly characterized by their local prominence, which motivates comparing each planner score with the scores in its immediate neighbourhood; we will denote this 8-cell neighbourhood (excluding x itself) by $\mathcal { N } _ { 1 } ( \times )$ . We then construct the binary neighbourcomparison mask to be

$$
M ( \mathbf { x } ) = \mathbf { 1 } \left[ \frac { 1 } { 8 } \sum _ { \mathbf { y } \in \mathcal { N } _ { 1 } ( \mathbf { x } ) } \mathbf { 1 } _ { \overline { { S } } ( \mathbf { x } ) > \overline { { S } } ( \mathbf { y } ) } \geq 0 . 7 \right] ,\tag{70}
$$

which thus retains cells whose scalar softmax-average score exceeds at least 70% of the scores in their local neighbourhood, excluding cells with neighbourhoods outside the grid boundary. The idea is that this mask will strip away enough of the score-field features to leave just the central ridges of the lightcone branches, which can then be compared with the null directions of Eq. (65).

Branch extraction is then performed row by row on this binary mask, where the efective mask threshold is $M ( \times ) > 0 . 5$ , so only cells with $M ( \times ) = 1$ are active, and the lightcone centre is fixed at $( x _ { c } , t _ { c } ) = ( 0 , 0 )$ . For each retained time row $t _ { j }$ , the active cells are divided into the left and right regions $x < x _ { c }$ and $x > x _ { c } ,$ with rows suficiently close to $t _ { c }$ excluded from the fit. Within each side, the active cells are divided into contiguous runs, so if a run contains N cells with mask values $p _ { k }$ , it is assigned the score

$$
S = \left( \sum _ { k \in \mathrm { r u n } } p _ { k } \right) { \sqrt { N } } .\tag{71}
$$

Since the present mask is binary, this reduces to $S =$ $N ^ { 3 / 2 }$ , so the widest contiguous active run on each side is retained. A representative branch coordinate is then obtained from the peak cell of the selected run, and at most one left and one right branch point are retained for each time row. The upper and lower lightcone arms are both included through the positive and negative time rows, and the resulting left and right branch points are combined using

$$
\Delta t _ { i } = t _ { i } - t _ { c } ,\tag{72}
$$

$$
\Delta x _ { i } = x _ { i } - x _ { c } ,\tag{73}
$$

which are then compared with the null directions of Eq.   
(65) to compute the components of the metric tensor.

The results are shown in Fig. 6.

In the coordinate ordering (t, x), the resulting coeficients obtained from the five-agent averaged field are assembled into the metric representative

$$
g _ { \mathrm { f i t } } = \left( { 7 . 7 8 7 \times 1 0 ^ { - 8 } } \begin{array} { c } { { 7 . 7 8 7 \times 1 0 ^ { - 8 } } } \\ { { 7 . 7 8 7 \times 1 0 ^ { - 8 } } } \end{array}  \right) ,\tag{74}
$$

whose eigenvalues are approximately (−0.999878, 1) with determinant −0.999878 and corresponding null slopes $v _ { \pm } \simeq ( - 0 . 9 9 9 9 3 9 , 0 . 9 9 9 9 3 9 )$ , thus yielding a metric with Lorentzian signature. For comparison, flat $( 1 + 1 )$ dimensional Minkowski spacetime has $g _ { t t } = - 1 , g _ { t x } = 0 ,$ $g _ { x x } \ = \ 1$ , and $v _ { \pm } = \pm 1$ . The reconstructed time component difers from its exact value by approximately $1 . 2 2 \times 1 0 ^ { - 4 }$ , the fitted of-diagonal contribution is negli gible relative to the diagonal entries, and the null slopes difer from ±1 by approximately $6 . 1 0 \times 1 0 ^ { - 5 \mathrm { ~ 3 ~ } }$ . Within the conformal convention $g _ { x x } = 1$ , the result therefore recovers a representative of the conformal class that closely approximates the Minkowski metric in these coordinates, demonstrating that the learned planner representation retains suficient geometric information to approximately reconstruct the causal structure of the underlying spacetime.

## VI. DISCUSSION AND OUTLOOK

The results from Section V provide a clear picture where an RL agent given a quantum information task embedded in a spacetime learns to accomplish that task while adhering to the geometry of the underlying spacetime. Through construction of a rich internal model of the en vironment that contains enough geometric and causal information, the agent is able to quantitatively reconstruct the Lorentzian metric of spacetime. Not only is the spacetime information implicitly hidden in the RL agent’s environment feedback—-in addition to the fact that the RL agent is initialized with no information about what spacetime is—-but the architecture of the RL agent also scrambles that information into an internal latent representation, disconnected with the exterior spacetime grid. This connects the present construction to work on intuitive physics, where agents can acquire internal representations supporting physical prediction without explicitly evaluating any equations of motion [31, 61, 62]. In particular, predictive models have been shown to acquire identifiable physical concepts from experience despite the absence of them in the explicit output of the prediction task [31].

![](images/bdf2ec15968a528b2f67a5e5dcd4aec766ae9b632b32e3a32b341ff5dbed4014.jpg)

![](images/f47258e8a5f45d5b864b394c4983c5ebbe642c65341cf6a7c246eaf6c51292d2.jpg)  
FIG. 6. Reconstruction of the conformal metric from the scalar softmax-average planner score ${ \overline { { S } } } .$ Left: local neighbourcomparison mask obtained by retaining cells whose $\overline { S }$ value exceeds at least 70% of the values in a radius-1 neighbourhood. Right: quadratic form displaying Lorentzian signature fitted to the extracted lightcone branches with the conformal convention $g _ { x x } = 1$ . The resulting null directions closely reproduce those of conformally flat (1 + 1)-dimensional Minkowski spacetime.

This suggests an operational perspective on spacetime concept formation, where an RL agent does not need to possess an explicit symbolic representation of a Lorentzian manifold in order for spacetime structure to organize its behaviour. Instead, regularities in its interactions can become encoded as relations among internal states, predictions, and actions. In the present model the planner never explicitly evaluates $g _ { \mu \nu } \Delta x ^ { \mu } \Delta x ^ { \nu } = 0$ when choosing an action, yet the memory supporting those choices becomes sensitive to operational regularities from which this null condition can subsequently be recovered. In this restricted sense, the learned representation provides a primitive model of a spacetime concept as an action-relevant organization of experience that implicitly captures causal relations, analogous to the metric tensor. This is consistent with the view that memory goes beyond functioning as just an inert storage device, but plays a more active role as a resource for generating future behaviour [63–65].

This point of view does highlight an important distinction however, which is that the RL agent does not carry the actual metric tensor object in its memory. As we saw earlier, it’s memory acquires a geometric flavour from learning, and the metric tensor is subsequently extracted using external tools supplied when studying it, such as lightcone masking and conformal gauge fixing. While using such geometrically informed post-processing procedures is unsatisfactory compared to the larger vision of explicitly emergent causal or metric structure in the RL agent’s memory, we view this work as merely a first step towards this setting. A stronger realization of this programme could of course make geometric inference itself part of the learned world model, for instance as a learned sub-network of the RL agent’s internal model.

Minkowski spacetime also provides a deliberately simple test of the above ideas that gives us increased clarity by making every step of the reconstruction process inspectable, especially the memory of the RL agent. Of course its recovery — particularly in the (1 + 1)- dimensional case — is not in and of itself surprising, since the causal structure is known analytically and is already reflected in the known field correlations from which the detector data are generated. One should realize however, that the architecture and problem setup are agnostic to the underlying geometry of the data, so in the general setting they can still be employed to learn the structure even when it is not known in advance, meaning that the same strategy can be applied when detector generated quantum information is available but the spacetime from which it originates is not. More robust tests of the reconstruction under coordinate changes and diferent detector smearings should be undertaken as well before making this step. This would substantially strengthen the evidence that the learned relations represent spacetime structure as opposed to contingent properties of the chosen coordinates or probes.

The question of how far we can take this interpretation will require several extensions. First, quantum mutual information is not a pure measure of signalling as it contains both causally mediated and state-dependent correlations, including correlations at spacelike separation. Future agents should thus receive several complementary operational quantities that can tease these diferences apart, such as signalling estimators or spin correlations, and more generally observables that are separately sen sitive to commutator and anticommutator contributions.

These may then allow for causal propagation and scale information to be distinguished and could provide the additional operational information required to reconstruct more than just the conformal structure. The setting should likewise be extended to (3 + 1)-dimensional and curved spacetimes, where the local causal structure varies across the environment and cannot be identified with two globally straight lightcone branches like in Minkowski.

Finally, one can consider more challenging or practical quantum information tasks as the learning objective, since maximizing a scalar function of mutual information provides a nice transparent proof of principle, but a more compelling test would place the RL agent in a task for which learning spacetime structure is actually necessary for completing the task. Quantum communication or adaptive entanglement distribution through detector networks could be natural and interesting possibilities, which may also require the RL agent to utilize longerhorizon planning that would test how it progresses from implicitly exploiting causal regularities to constructing increasingly general representations of the spacetime in which it acts. The quantum information task studied here shows how the causal structure of Minkowski spacetime can emerge in the memory of an agent, opening the doors to the more ambitious question of whether agents can autonomously discover an explicit spacetime representation from operational experience alone.

## ACKNOWLEDGMENTS

The authors would like to thank Hans J. Briegel for initial discussions leading to the conception of the project, as well as for ongoing discussions and feedback on drafts. This research was funded in whole or in part by the European Union (ERC Advanced Grant, QuantAI, No. 101055129) and the Austrian Science Fund (FWF) [SFB BeyondC F7102, 10.55776/F7102]. The views and opinions expressed in this article are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council — neither the European Union nor the granting authority can be held responsible for them. For open access purposes, the authors have applied a CC BY public copyright license to any author-accepted manuscript version arising from this submission. T. R. Perche is hosted at the Physics Department of Stockholm University through a Wenner-Gren Fellowship and is thankful for partial financial support from the Olle Engkvist Foundation (no.225- 0062).

[1] C. M. Will, The confrontation between general relativity and experiment, Living Reviews in Relativity 17, 4 (2014).

[2] R. M. Wald, General Relativity (University of Chicago Press, Chicago, 1984).

[3] J. Ehlers, F. A. E. Pirani, and A. Schild, Republication of: The geometry of free fall and light propagation, General Relativity and Gravitation 44, 1587 (2012), originally published in 1972.

[4] S. A. Basri, Operational foundation of einstein’s general theory of relativity, Reviews of Modern Physics 37, 288 (1965).

[5] J. L. Synge, Relativity: The General Theory (North-Holland, Amsterdam, 1960).

[6] V. Perlick, On the radar method in general-relativistic spacetimes, in Lasers, Clocks and Drag-Free Control: Exploration of Relativistic Gravity in Space, edited by H. Dittus, C. L¨ammerzahl, and S. G. Turyshev (Springer, Berlin, Heidelberg, 2008) pp. 131–152.

[7] S. W. Hawking, A. R. King, and P. J. McCarthy, A new topology for curved space-time which incorporates the causal, diferential, and conformal structures, Journal of

The authors would also like to disclose the use of AI tools (OpenAI Codex with GPT-5.5) for the creation of Fig. 1 as well as code generation for obtaining plots, optimizing and debugging code, and automation of standard and publicly available code blocks. All ideas, results, and scientific conclusions originate with the authors. We have verified everything thoroughly and take full responsibility for everything contained in the manuscript.

Mathematical Physics 17, 174 (1976).

[8] D. B. Malament, The class of continuous timelike curves determines the topology of spacetime, Journal of Mathematical Physics 18, 1399 (1977).

[9] M. Saravani, S. Aslanbeigi, and A. Kempf, Spacetime curvature in terms of scalar field propagators, Physical Review D 93, 045026 (2016).

[10] T. R. Perche and E. Mart´ın-Mart´ınez, Geometry of spacetime from quantum measurements, Phys. Rev. D 105, 066011 (2022).

[11] R. Haag, Local Quantum Physics: Fields, Particles, Algebras, 1st ed. (Springer Publishing Company, Incorporated, 2012).

[12] C. Cao, S. M. Carroll, and S. Michalakis, Space from hilbert space: Recovering geometry from bulk entanglement, Physical Review D 95, 024031 (2017).

[13] A. Kempf, Replacing the notion of spacetime distance by the notion of correlation, Frontiers in Physics 9, 655857 (2021).

[14] G. Franzmann, S. M. D. Jovancic, and M. Lawson, Relative distance of entangled systems in emergent spacetime scenarios, Physical Review D 107, 066008 (2023).

[15] W. G. Unruh, Notes on black-hole evaporation, Phys. Rev. D 14, 870 (1976).

[16] B. S. DeWitt, Quantum gravity: the new synthesis, in General Relativity: An Einstein Centenary Survey, edited by S. W. Hawking and W. Israel (Cambridge University Press, Cambridge, 1979) pp. 680–745.

[17] C. J. Fewster and R. Verch, Quantum fields and local measurements, Communications in Mathematical Physics 378, 851 (2020).

[18] D. Grimmer, B. d. S. L. Torres, and E. Mart´ın-Mart´ınez, Measurements in QFT: Weakly coupled local particle detectors and entanglement harvesting, Physical Review D 104, 085014 (2021).

[19] J. Louko and A. Satz, Transition rate of the unruh–DeWitt detector in curved spacetime, Class. Quantum Gravity 25, 055012 (2008).

[20] E. Mart´ın-Mart´ınez, Causality issues of particle detector models in qft and quantum optics, Physical Review D 92, 104019 (2015).

[21] E. Mart´ın-Mart´ınez, T. R. Perche, and B. d. S. L. Torres, Broken covariance of particle detector models in relativistic quantum information, Physical Review D 103, 025007 (2021).

[22] L. C. B. Crispino, A. Higuchi, and G. E. A. Matsas, The unruh efect and its applications, Reviews of Modern Physics 80, 787 (2008).

[23] E. Mart´ın-Mart´ınez, A. R. H. Smith, and D. R. Terno, Spacetime structure and vacuum entanglement, Phys. Rev. D 93, 044001 (2016).

[24] Y.-Z. You, Z. Yang, and X.-L. Qi, Machine learning spatial geometry from entanglement features, Physical Review B 97, 045153 (2018).

[25] K. Hashimoto, S. Sugishita, A. Tanaka, and A. Tomiya, Deep learning and the ads/cft correspondence, Physical Review D 98, 046019 (2018).

[26] K. Hashimoto, S. Sugishita, A. Tanaka, and A. Tomiya, Deep learning and holographic qcd, Physical Review D 98, 106014 (2018).

[27] K. Hashimoto, K. Matsuo, M. Murata, G. Ogiwara, and D. Takeda, Machine-learning emergent spacetime from linear response in future tabletop quantum gravity experiments, Machine Learning: Science and Technology 6, 015030 (2025).

[28] B. Ahn, H.-S. Jeong, K.-Y. Kim, and K. Yun, Holographic reconstruction of black hole spacetime: machine learning and entanglement entropy, arXiv preprint arXiv:2406.07395 (2024), arXiv:2406.07395 [hep-th].

[29] D. Grimmer, I. Melgarejo-Lermas, J. Polo-G´omez, and E. Mart´ın-Mart´ınez, Decoding quantum field theory with machine learning, Journal of High Energy Physics 2023, 31 (2023).

[30] S. Shrapnel, G. Milburn, and M. Kewming, Quantumenhanced agents: causal machine learning in a quantum world, in Emerging Topics in Artificial Intelligence 2020, Vol. 11469 (SPIE, 2020) p. 114690Y.

[31] L. S. Piloto, A. Weinstein, P. Battaglia, and M. Botvinick, Intuitive physics learning in a deeplearning model inspired by developmental psychology, Nature Human Behaviour 6, 1257 (2022).

[32] P. A. LeMaitre, https://github.com/AntiPhoton47/ Learning-Spacetime-from-Quantum-Info.

[33] S. Takagi, Vacuum Noise and Stress Induced by Uniform Acceleration: Hawking-Unruh Efect in Rindler Manifold of Arbitrary Dimension, Prog. Theor. Phys. Supp. 88, 1

(1986).

[34] S. Schlicht, Considerations on the unruh efect: Causality and regularization (2003).

[35] J. de Ram´on, M. Papageorgiou, and E. Mart´ın-Mart´ınez, Causality and signalling in non-compact detector-field interactions, Physical Review D 108, 045015 (2023).

[36] A. Pozas-Kerstjens and E. Mart´ın-Mart´ınez, Harvesting correlations from the quantum vacuum, Phys. Rev. D 92, 064042 (2015).

[37] A. Pozas-Kerstjens and E. Mart´ın-Mart´ınez, Entanglement harvesting from the electromagnetic vacuum with hydrogenlike atoms, Phys. Rev. D 94, 064074 (2016).

[38] P. A. LeMaitre, Harvesting contextuality from the vacuum, Physical Review D 113, 045001 (2026).

[39] E. Martin-Martinez, D. Aasen, and A. Kempf, Processing quantum information with relativistic motion of atoms, Phys. Rev. Lett. 110, 160501 (2013).

[40] D. E. Bruschi, A. Dragan, A. R. Lee, I. Fuentes, and J. Louko, Relativistic motion generates quantum gates and entanglement resonances, Phys. Rev. Lett. 111, 090504 (2013).

[41] E. Mart´ın-Mart´ınez and C. Sutherland, Quantum gates via relativistic remote control, Phys. Lett. B 739, 74–82 (2014).

[42] E. Aspling and M. Lawler, Universal quantum computing with field-mediated unruh–dewitt qubits (2024), arXiv:2402.10173.

[43] P. A. LeMaitre, T. R. Perche, M. Krumm, and H. J. Briegel, Universal quantum computer from relativistic motion, Phys. Rev. Lett. 134, 190601 (2025).

[44] R. Lopp and E. Mart´ın-Mart´ınez, Quantum delocalization, gauge, and quantum optics: Light-matter interaction in relativistic quantum information, Physical Review A 103, 013703 (2021).

[45] T. R. Perche, B. Ragula, and E. Mart´ın-Mart´ınez, Harvesting entanglement from the gravitational vacuum, Physical Review D 108, 085025 (2023), arXiv:2210.14921 [gr-qc, physics:hep-th, physics:quant-ph].

[46] A. Valentini, Non-local correlations in quantum electrodynamics, Phys. Lett. A 153, 321 (1991).

[47] B. Reznik, Entanglement from the vacuum, Found. Phys. 33, 167 (2003).

[48] R. H. Jonsson, K. Ried, E. Martin-Martinez, and A. Kempf, Transmitting qubits through relativistic fields, Journal of Physics A: Mathematical and Theoretical 51, 485301 (2018).

[49] P. Simidzija, A. Ahmadzadegan, A. Kempf, and E. Mart´ın-Mart´ınez, Transmission of quantum information through quantum fields, Physical Review D 101, 036014 (2020).

[50] A. G. S. Landulfo, Nonperturbative approach to relativistic quantum communication channels, Phys. Rev. D 93, 104019 (2016).

[51] T. R. Perche, Closed-form expressions for smeared bidistributions of a massless scalar field: Nonperturbative and asymptotic results in relativistic quantum information, Phys. Rev. D 110, 025013 (2024).

[52] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. (MIT Press, 2018).

[53] R. Bellman, Dynamic Programming (Princeton University Press, 1957).

[54] C. J. C. H. Watkins, Learning from Delayed Rewards, Ph.D. thesis, King’s College, Cambridge (1989).

[55] C. J. C. H. Watkins and P. Dayan, Q-learning, Machine

Learning 8, 279–292 (1992).

[56] G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of Control, Signals and Systems 2, 303–314 (1989).

[57] K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators, Neural Networks 2, 359–366 (1989).

[58] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, Playing atari with deep reinforcement learning, arXiv preprint arXiv:1312.5602 (2013).

[59] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, Human-level control through deep reinforcement learning, Nature 518, 529–533 (2015).

[60] L.-J. Lin, Self-improving reactive agents based on re-

inforcement learning, planning and teaching, Machine Learning 8, 293–321 (1992).

[61] P. W. Battaglia, J. B. Hamrick, and J. B. Tenenbaum, Simulation as an engine of physical scene understanding, Proceedings of the National Academy of Sciences 110, 18327 (2013).

[62] B. M. Lake, T. D. Ullman, J. B. Tenenbaum, and S. J. Gershman, Building machines that learn and think like people, Behavioral and Brain Sciences 40, e253 (2017).

[63] H. J. Briegel and G. De las Cuevas, Projective simulation for artificial intelligence, Scientific Reports 2, 400 (2012), arXiv:1104.3787.

[64] P. A. LeMaitre, M. Krumm, and H. J. Briegel, Multi-excitation projective simulation with a manybody physics inspired inductive bias, arXiv:2402.10192 10.48550/arXiv.2402.10192 (2024).

[65] H. J. Briegel and T. M¨uller, Projective Simulation in Action: Quantum-Mechanical Perspectives on the Problem of Agency (Springer Nature, 2025).

# APPENDICES

## Appendix A: Supplementary Internal Diagnostics

Figure 5 in Section V displayed some of the important diagnostics for assessing the quality of the RL agent’s internal model, and here we will provide more complementary diagnostics to those plots. These remaining plots shown in Fig. 7 provide a more detailed look at the properties of the model components, and their role is to solidify that the score generated by the planner, which is used for the metric extraction, is produced by properly functioning internal predictive models along with the encoder/decoder.

Plot (a) displays the argmax action map and records how the planner partitions the grid into local action regions, clearly displaying the lightcone structure on the diagonals but also drawing attention to diferent local action domains. Plots (b) and (c) give the decoded coordinates which unambiguously show that the x and t coordinate organization is preserved by the encoder and decoder. Plot (d) gives the latent norm, which remains finite and spatially structured over the grid, and indicates that the latent state components are better shaped as we move closer to the goal state. Plots (e) and (f) display the action-conditioned dynamics error and the mean predicted latent step norm, both of which have the lightcone structure somewhat visible in them, with the first indicating that action-conditioned dynamics becomes slightly less reliable the further away from the goal that we move, and the second showing that the imagined latent displacement across the grid remains fairly controlled. Plots (g)—(i) show that the next-state reconstruction errors remain small away from the extreme boundary region. Finally, plots (j)—(l) compare three reward model projections: the mean over actions, the maximum over actions, and the reward assigned to the plannerselected action. Their common lightcone-shaped support is consistent with the reward range reported in Sec. V and with the planner-step reward shown in Fig. 5. Thus, all plots are consistent with and support the narrative that the RL agent learned a useful representation of the spacetime to recover the Lorentzian quadratic form in Eq. (74).

![](images/8e010db763ed8cb35a1d05c75a292e3bc6e9f677ecacf0ceb731b1395b38cf72.jpg)

![](images/2b94f6a48a3e16be8deb711d973a1b6f2b9dda0e3eaa0ec8aa34faeae4c1e7b2.jpg)

![](images/414718e6b7f6ff8f8ceace876106b8efc2ea144f55e0c33b674e676d94c19ad5.jpg)

![](images/77793ea5c7d3fc8d59e0636c59bbd5f0943fb01f50a4ecdcb6484744587023f2.jpg)

![](images/4f106d235fa615d5219c779b2149266cfa805860225653d7deba37e94a2924c2.jpg)

![](images/74b6a3d8a37e7cb1d567a992188c9f51dabee12b8323c5a5898b2d3cfc01497a.jpg)

![](images/1805dea69e1608943670fb1d512c53f54ef6ffb37ca9d5ca4b8b235b58cdf806.jpg)

![](images/b2d9c14c81346a68c66a777956ca657598f5f8ae32ec74d2f0355733e8cddd56.jpg)

![](images/00b1c9046ce4e395cfb38ad8c9d8ba159ef85fc75a8d4c4194d007463ad2a84e.jpg)

![](images/ccfd1df0f164f54d9c716356479051bee58f97cd505e190458ccb55326c2800d.jpg)

![](images/12dff0d9c84f2fed4bd1057702d2038b3d05a3bcdb96f0bdaa248e9e5a47a94c.jpg)

![](images/806020e91d1ca5ea2c6d2bf8e5e521312808bd6c587f13e0c85bfe46f5254d02.jpg)  
FIG. 7. Supplementary internal diagnostics for the trained planner. Plot (a) shows the argmax action splitting into each action direction; plots (b) and (c) show the decoded coordinates; plot (d) shows the latent norm; plot (e) shows the dynamics error; plot (f) shows the mean predicted latent step norm; plots (g)—(i) show next-state reconstruction errors; and plots (j)—(l) show reward model projections.