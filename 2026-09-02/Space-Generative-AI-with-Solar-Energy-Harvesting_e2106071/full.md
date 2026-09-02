# Space Generative AI with Solar Energy Harvesting

Jierui Zhang, Jianhao Huang, Zhanwei Wang, and Kaibin Huang

Abstract—Satellites are emerging as promising platforms for extending generative artificial intelligence (AI) services to remote areas lacking terrestrial infrastructure. However, deploying space generative AI is fundamentally constrained by the limited and time-varying onboard energy supplied by solar energy harvesting (EH). This paper presents a framework for solar-powered space generative AI in which a satellite receives a user prompt, executes a diffusion-based image-generation model, and downlinks the compressed result within a strict time window. To design this framework, we identify fundamental computation–communication (C<sup>2</sup>) trade-offs governed by the shared harvested-energy budgets. Specifically, increasing the number of generation steps improves intrinsic image quality but depletes energy and time available for downlink transmission, whereas prioritizing communication guarantees reliable delivery but sacrifices semantic quality. To balance these trade-offs and maximize end-to-end (E2E) generative performance, we exploit the predictable solar-EH dynamics induced by deterministic orbital motion and develop a joint C<sup>2</sup> resource-optimization framework using a tractable two-step approach. First, we characterize the maximum downlink throughput for a fixed generation depth under continuous solar EH. This establishes a separation principle that decouples waiting-time selection from optimal transmit-power control. Next, building on this principle, we formulate a joint $\mathbf { C } ^ { 2 }$ utility-maximization problem and derive a closed-form, low-complexity step-selection policy in the dominant constant-power regime. Extensive experiments under realistic orbital dynamics demonstrate that the proposed policy dynamically balances generation quality and transmission reliability. This yields significant E2E performance gains over static computation- and communication-centric baselines across diverse solar EH states.

Index Terms—Space computing network, solar energy harvest ing, generative AI, space AI.

## I. INTRODUCTION

With rapid advances in generative artificial intelligence (AI) such as high-fidelity image generation and complex reasoning, supporting globally accessible generative-AI services is becoming a critical objective for future networks [1]–[4]. However, delivering generative-AI services to remote and underserved areas remains challenging due to the lack of terrestrial AI infrastructure. Although satellites have traditionally served as passive relays connecting remote users to ground networks, this relay-based paradigm may incur significant latency. Enabled by increasing onboard computation capability, satellites are emerging as active AI-service providers for ubiquitous generative-AI service provisioning beyond the coverage of terrestrial networks [5]–[9]. Meanwhile, satellites can harvest solar energy directly in orbit, avoiding atmospheric attenuation and many ground-level obstructions, thereby supporting onboard tasks through solar energy harvesting (EH). These advantages have motivated two related architectural visions: space data centers, which deploy large-scale computing infrastructure in orbit, and space-ground integrated computing power networks, which coordinate computing resources across satellites, terrestrial edge nodes, and cloud data centers [10]–[12]. A recent example is StarCloud, which launched a prototype orbital computing facility equipped with an H100 GPU for onboard model training and inference, further indicating the feasibility of executing AI workloads in orbit [13]. However, realizing practical space generative AI still requires overcoming a substantial challenge: onboard AI inference and downlink transmission typically rely on timevarying energy supplied by solar EH, while computation and communication are tightly coupled through the shared onboard energy budget. Although sun-synchronous orbits can mitigate solar-energy variation in some cases, such favorable orbits cannot accommodate all space-AI deployments due to limited orbital resources and mission-specific coverage requirements. Hence, space AI systems must also be designed for general orbital conditions, where solar EH can be time-varying. This paper investigates the joint management of computation and communication $( \mathbf { C } ^ { 2 } )$ for space generative AI, with the goal of maximizing end-to-end (E2E) performance under solar-EH constraints.

The deployment of generative AI is inherently energyintensive due to its heavy computational workload, posing significant challenges for energy-constrained onboard AI execution [14]. Unlike terrestrial AI systems that typically have access to stable power-grid infrastructure, orbital AI must rely on limited onboard energy resources. To sustain off-grid operations, EH has emerged as a key enabling technology and has attracted extensive research attention, particularly in wireless communications [15]. In satellite systems, solar EH is the primary energy source, but the harvested power is not a static budget. Instead, it varies continuously with orbital motion, solar incidence angle, panel orientation, and sunlight-eclipse transitions [16]–[18]. Although the satelliteground visibility window may be much longer than a single service request, an interactive generative-AI task typically imposes a much shorter execution window to ensure a timely response [19]. This short window limits both the time available for energy accumulation and the duration available for onboard computation and downlink transmission. As a result, the satellite cannot simply wait until sufficient solar energy is harvested, but must complete waiting, computation, and communication within a tight service deadline. During this window, the available onboard energy is strongly determined by the satellite’s orbital position and solar-EH-induced battery state. Therefore, completing such a task requires careful allocation of limited harvested energy and time between onboard generation and downlink transmission.

This allocation problem is particularly challenging for space generative AI because computation and communication jointly determine the final received content quality. Consider a textto-image service where a ground device uploads a lightweight prompt to a satellite, the satellite generates an image onboard using a diffusion model [20], [21], and the generated image is then compressed and downlinked to the ground device. In this pipeline, computation and communication draw from the same harvested-energy buffer. Increasing the number of denoising steps generally improves the intrinsic quality of the generated image, but it consumes more energy and leaves less energy for downlink transmission. Conversely, reserving more energy for transmission increases the delivery capacity and can reduce compression distortion, but it forces the generative process to use fewer denoising steps, which degrades semantic quality. $\mathbf { A } s$ a result, a computation-centric policy synthesizes a high-quality image that cannot be faithfully delivered, while a communication-centric policy reliably delivers an image whose intrinsic quality is poor. Therefore, maximizing the E2E generation quality requires task-aware allocation of energy between onboard generation and downlink transmission under solar-EH-induced causality constraints.

Existing studies provide useful foundations, but they do not fully address the above generate-and-deliver trade-off. A first line of work is EH communications, which studies transmission scheduling under energy-causality constraints. Classical results characterize optimal throughput-maximizing policies through methods such as directional water filling and save-then-transmit strategies [22]–[25]. These works establish important principles for energy-causal transmission, but they typically allocate harvested energy solely to communication, without considering computation that may compete for the same energy buffer. A second related line considers energyaware satellite systems. Several studies have investigated satellite energy management, sunlight-aware task scheduling, and battery-aware edge computing under orbital energy constraints [16]–[18]. These works highlight the importance of orbital energy dynamics, but they mainly focus on routing, task scheduling, or generic computing workloads. They do not model the task-specific quality behavior of generative AI, where the computation depth directly affects the semantic quality of the output. Finally, recent studies have explored joint computation and communication resource allocation, as well as broader sensing–communication–computation integration [26]. Some works minimize download time or information freshness by jointly optimizing onboard processing and transmission [27]–[29], while others focus on energy efficiency, computation offloading, compression, or network throughput [30]–[37]. However, most of these formulations treat computation as a fixed workload or a generic offloading/compression task. They do not capture a key property of diffusion-based generation: the number of denoising steps is a controllable quality knob that should be jointly adapted with downlink transmission. Moreover, solar EH is rarely considered in generative-service-oriented satellite designs. Consequently, existing EH communication and satellite $C ^ { \bar { 2 } }$ frameworks are insufficient for optimizing E2E generative performance under solar EH.

The key challenges of optimizing such a framework arise from two factors: (1) the time-varying profiles of both the wireless channel and solar EH, and (2) the coupling between onboard generation and subsequent downlink transmission under the solar-EH profile. For the first challenge, deterministic orbital motion and geometric relationships make these profiles predictable over the task execution window and provide tractable mathematical models, thereby enabling proactive resource allocation. For the second challenge, onboard generation and downlink transmission jointly determine the E2E performance, requiring the system to manage generation quality, waiting time, and communication throughput under a shared solar-EH-powered energy buffer, rather than merely scheduling transmit power as in conventional EH communication. To study this joint $C ^ { 2 }$ optimization for space generative AI, we decompose the E2E performance into two coupled components: onboard generation quality and downlink throughput. We first fix the generation step and optimize the waiting time and transmit power to characterize the maximum achievable downlink throughput under continuous energy causality. Building on this characterization, we then formulate a joint $C ^ { 2 }$ utility that balances onboard generation quality and downlink throughput, and derive a closed-form, low-complexity step-selection rule in the dominant constantpower regime. Specifically, the key contributions and findings are summarized as follows:

• Communication Throughput Maximization under Solar EH: We fix the generation step and investigate a communication-throughput maximization problem. This problem not only serves as the inner subproblem of the subsequent joint $C ^ { 2 }$ optimization, but also becomes critical when the generated content has a large downlink data volume. Unlike classical EH communication, which provides general structural solutions for arbitrary EH profiles, our problem involves an orbital solar-EH profile and a waiting-time decision introduced by onboard task execution. To solve the problem, we instantiate a tractable, representative solar-EH model based on orbital geometry, prove a separation principle between waitingtime selection and transmit-power control, and derive the optimal energy-feasible transmit-power policy using the lower convex envelope (LCE) principle.

• Joint $\mathbf { C } ^ { 2 }$ Optimization under Solar EH: Beyond the fixed-step communication-throughput optimization, we further consider the general case where onboard generation quality and downlink throughput must be jointly balanced. Specifically, we approximate the relationship between onboard generation quality (measured by $C L I P$ score [38], [39]) and generation steps, and then formulate a joint $C ^ { 2 }$ utility maximization problem over the generation step, waiting time, and transmit-power policy. This problem is challenging because these control variables are tightly coupled through the shared solar-EH-powered energy buffer. By decomposing the problem with respect to the generation step, the inner subproblem is connected to the throughput-maximization problem derived above. In the dominant constant-power regime, this decomposition yields a reduced single-variable problem, for which we derive a closed-form, low-complexity step-selection rule using the Lambert W function. The resulting rule enables lightweight onboard adaptation and reveals how the optimal generation step changes with the solar-EH state and system parameters.

• Experimental Results: We conduct comprehensive experiments to evaluate the proposed space generative AI framework under realistic low-Earth-orbit (LEO) parameters. The results show that the proposed joint optimization scheme adapts between computation-centric and communication-centric schemes according to the solar-EH state. It approaches the more suitable extreme policy in energy-scarce or energy-abundant regimes, while outperforming both extremes in moderate-energy regimes through balanced onboard generation and downlink transmission. Overall, it achieves more robust E2E generation performance than fixed extreme allocation schemes.

The remainder of this paper is organized as follows. Section II introduces the models and metrics. Section III solves the communication throughput maximization problem. Section IV investigates a joint $C ^ { 2 }$ optimization problem and derives the closed-form solution. Experimental results are provided in Section V, followed by concluding remarks in Section VI.

## II. SYSTEM MODEL

Consider a space generative AI framework powered by solar EH, as shown in Fig. 1. In this framework, a static ground device offloads prompt-based generation tasks to a satellite. Then, the solar-EH-powered satellite completes a sequence of onboard operations within a short task execution window of duration $T ,$ including waiting, content generation, and result transmission. The solar EH model, onboard computation model, space-ground communication model, and performance metrics are detailed in the following subsections.

## A. Solar EH Model

In this subsection, we model satellite solar EH based on orbital geometry. Two Cartesian coordinate systems, the orbital reference frame $( \mathcal { F } _ { o } )$ and the satellite body frame $( \mathcal { F } _ { b } )$ [40], are as shown in Fig. 1(a).

• Orbital Reference Frame $( \mathcal { F } _ { o } ) \colon$ The origin is located at the center of the Earth. The $Y _ { o }$ -axis is orthogonal to the orbital plane. The $X _ { o } \mathrm { - a x i s }$ points towards the projection of the sun vector onto the orbital plane, and the $Z _ { o } \mathrm { - a x i s }$ completes the right-handed system. The sun vector is

$$
\mathbf { S } _ { o } = [ \cos \beta , \sin \beta , 0 ] ^ { T } ,\tag{1}
$$

where $\beta$ denotes the solar elevation angle.

• Satellite Body Frame $( \mathcal { F } _ { b } ) \colon$ The origin is the satellite’s center of mass. We adopt a standard nadir-pointing attitude [41], where z points towards the Earth (nadir), y is aligned with the negative orbit normal (cross-track) and $x _ { b }$ is aligned with the velocity vector (along-track).

The angular velocity ω of the satellite is determined by Kepler’s third law, i.e., $\begin{array} { r } { \omega = \sqrt { \frac { G M } { ( R _ { \mathrm { E } } + H ) ^ { 3 } } } } \end{array}$ , where $G$ is the gravitational constant, M is the mass of the Earth, $R _ { \mathrm { E } }$ is the Earth radius, and H is the orbital altitude [42]. Consider a solar panel rigidly mounted on the satellite body, with unit normal vector $\mathbf { n } = [ n _ { x } , n _ { y } , n _ { z } ] ^ { T } \in \mathbb { R } ^ { 3 }$ expressed in ${ \mathcal { F } } _ { b } .$ . Assume that t = 0 corresponds to the orbital noon and that $\mathbf { n } = [ 0 , 0 , - 1 ] ^ { T }$ (i.e., the anti-nadir direction); then the solar-EH profile (the instantaneous harvested power) $\mathrm { i s } ^ { 1 }$

$$
P ( t ) = \left\{ \begin{array} { l l } { P _ { 0 } \cos ( \omega t ) , } & { 0 \leq t \leq \frac { \pi } { 2 \omega } , } \\ { 0 , } & { \frac { \pi } { 2 \omega } \leq t \leq \frac { 3 \pi } { 2 \omega } , } \\ { P _ { 0 } \cos ( \omega t ) , } & { \frac { 3 \pi } { 2 \omega } \leq t \leq \frac { 2 \pi } { \omega } . } \end{array} \right.\tag{2}
$$

Here, $P _ { 0 } = \eta \cdot \gamma \cdot A \cdot \cos \beta ,$ where $\eta$ is the power conversion efficiency, $\gamma$ is the solar irradiance per unit area, and A is the area of the solar panel [16]. Integrating (2) from the orbitalnoon reference yields the following representative cumulativeenergy profile:

$$
E ( t ) = \left\{ \begin{array} { l l } { \displaystyle \frac { P _ { 0 } } { \omega } \sin ( \omega t ) + E _ { 0 } , } & { 0 \leq t \leq \frac { \pi } { 2 \omega } , } \\ { \displaystyle \frac { P _ { 0 } } { \omega } + E _ { 0 } , } & { \displaystyle \frac { \pi } { 2 \omega } \leq t \leq \frac { 3 \pi } { 2 \omega } , } \\ { \displaystyle \frac { P _ { 0 } } { \omega } \big ( 2 + \sin ( \omega t ) \big ) + E _ { 0 } , } & { \displaystyle \frac { 3 \pi } { 2 \omega } \leq t \leq \frac { 2 \pi } { \omega } , } \end{array} \right.\tag{3}
$$

where $E _ { 0 } \geq 0$ denotes the initial energy at task arrival.

## B. Onboard Generative AI Model

Although generative AI supports various modalities, textto-image generation naturally fits the considered scenario: a power-limited ground user uploads only a lightweight prompt, while the solar-EH-powered satellite generates and downlinks the larger image result. Hence, we consider a typical text-toimage generation task, as described below.

1) Generative AI Model: We adopt a representative latent diffusion model (LDM) [20] as the generative backbone for text-to-image generation. The inference pipeline consists of three distinct phases as follows. First, a pre-trained text encoder τ<sub>θ</sub> processes the input prompt to produce conditional embeddings c. Second, initialized with random Gaussian noise $\mathbf { z } _ { n }$ in the latent space, the diffusion backbone iteratively refines the latent representation over n steps using the denoising diffusion implicit model (DDIM) scheduler conditioned on c [21]. Finally, the VAE decoder projects the clean latent $\mathbf { z } _ { 0 }$ back into the pixel space to generate the final image x.

![](images/8fc31557a56c3fc1e882439d499b5abdccb6b6145c4cfff99a1dc32418343751.jpg)  
(a) Space generative AI framework.

![](images/05d350e9a08d9d01e7f696e75497350887a54914a580d37cb2a81f1c55d85099.jpg)  
(b) Operations and protocol.  
Fig. 1: System model of space generative AI.

2) Computation Energy Consumption: We assume that, for each task, the computation power $P _ { \mathrm { c o m p } }$ is fixed. Let $\Omega _ { 1 }$ and $\Omega _ { 2 }$ denote the number offloating point operations $( { \mathrm { F L O P s } } )$ of each DDIM step and the FLOPs of remaining operations for a single inference (e.g., decoder of the VAE). Let ν denote the computation speed. Then, the computation time is

$$
t _ { 2 } = c _ { 1 } n + c _ { 2 } ,\tag{4}
$$

where $\begin{array} { r } { c _ { 1 } = \frac { \Omega _ { 1 } } { \nu } } \end{array}$ and $\begin{array} { r } { c _ { 2 } \ = \ \frac { \Omega _ { 2 } } { \nu } } \end{array}$ . Since the satellite may not possess sufficient energy to execute the task immediately upon reception, a waiting phase $t _ { 1 }$ may be required. Hence, the cumulative computation energy consumption is given by

$$
\begin{array} { r } { E _ { \mathrm { c o m p } } ( t ) = \left\{ \begin{array} { l l } { 0 , ~ } { { t } _ { \mathrm { r e q } } \leq t < t _ { \mathrm { r e q } } + t _ { 1 } , } \\ { P _ { \mathrm { c o m p } } ( t - t _ { \mathrm { r e q } } - t _ { 1 } ) , t _ { \mathrm { r e q } } + t _ { 1 } \leq t < a , } \\ { P _ { \mathrm { c o m p } } t _ { 2 } , ~ } { ~ } \quad ~ \right.} { a \leq t \leq b , } \end{array}   \end{array}\tag{5}
$$

where $a \triangleq t _ { \mathrm { r e q } } + t _ { 1 } + t _ { 2 }$ and $b \triangleq t _ { \mathrm { r e q } } + T$

## C. Space-ground Communication Model

We focus on downlink transmission because the promptuplink traffic is negligible. Consider a static ground device located on the Earth’s surface. We define the delivery capacity as the achievable number of bits delivered over the communication window $[ a , b ]$ . In this work, the satellite can exploit non-causal information because both the orbital trajectory and the space-ground channel evolution are predictable. Hence, the delivery capacity and the corresponding transmit power policy can be determined in advance (detailed in Section III). Then, the generated image x is compressed into $\mathbf { x } ^ { \prime }$ such that its data size does not exceed the delivery capacity. The compressed image is then transmitted according to the optimized power policy. The instantaneous channel gain $g ( t )$ is

$$
g ( t ) = G _ { \mathrm { t x } } G _ { \mathrm { r x } } L _ { \mathrm { F S P L } } L _ { \mathrm { A L } } ,\tag{6}
$$

where $G _ { \mathrm { t x } }$ and $G _ { \mathrm { r x } }$ denote the transmit and receive antenna gains, respectively [6], [43]. $\begin{array} { r } { L _ { \mathrm { F S P L } } = \left( \frac { \ell } { 4 \pi d ( t ) } \right) ^ { 2 } } \end{array}$ represents the large-scale free-space path-loss factor, where $\begin{array} { r } { \ell = \frac { c } { f _ { c } } } \end{array}$ is the downlink carrier wavelength, c is the speed of light, $f _ { c }$ is the carrier frequency, and $d ( t )$ is the time-varying distance between the satellite and the ground device. Note that $d ( t )$ is predictable based on the satellite’s orbital dynamics, thereby leading to a predictable channel. $L _ { \mathrm { A L } }$ accounts for additional environmental attenuation, such as rain fading. Small-scale fading is omitted, due to the nature of the considered remote area [6]. According to Shannon’s theorem [44], the instantaneous communication rate is given by

$$
r ( t ) = B \log _ { 2 } \left( 1 + \frac { P _ { \mathrm { c o m m } } ( t ) g ( t ) } { N _ { 0 } B } \right) ,\tag{7}
$$

where $P _ { \mathrm { c o m m } } ( t )$ is the transmit power, $N _ { 0 }$ is the noise power spectral density, and B is the bandwidth. Then, the cumulative communication energy consumption is given by

$$
\begin{array} { r } { E _ { \mathrm { c o m m } } ( t ) = \left\{ \begin{array} { l l } { 0 , } & { t _ { \mathrm { r e q } } \leq t < a , } \\ { \int _ { a } ^ { t } P _ { \mathrm { c o m m } } ( \tau ) \mathrm { d } \tau , } & { a \leq t \leq b . } \end{array} \right. } \end{array}\tag{8}
$$

## D. Performance Metrics

We aim to maximize the E2E generation quality, measured by the E2E CLIP score, i.e., the semantic similarity between the text prompt and the image received at the ground device [38], [39]. However, directly characterizing this E2E metric in closed form is challenging, as it depends jointly on the generative model, compression process, and downlink transmission. Therefore, we instead consider two tractable factors, corresponding to computation and communication, respectively, that jointly determine the E2E metric:

• Onboard CLIP score: This metric measures the semantic similarity between the text prompt and the onboardgenerated image, reflecting its intrinsic quality. Empirical results show that the CLIP score S increases with the number of DDIM steps n.

• Communication throughput: This metric is defined as the total number of bits delivered over the communication window [22], given by

$$
R = \int _ { a } ^ { b } r ( t ) { \mathrm { d } } t .\tag{9}
$$

A higher communication throughput allows a lower compression ratio and thus better preserves the quality of the generated image. It is related to waiting time, computation time, and transmit power policy.

(P2)

Remark 1. Under solar-EH constraints, these two factors exhibit an inherent $C ^ { 2 }$ trade-off: allocating more energy to onboard generation improves intrinsic image quality, but reduces the energy availablefor downlink transmission, and vice versa. This motivates the analytical design in the following two sections: Section III studies communication-throughput maximization whenfixing computation, while Section IVjointly optimizes $C ^ { 2 }$ for E2E performance maximization.

## III. COMMUNICATION THROUGHPUT MAXIMIZATION UNDER SOLAR EH

In this section, we maximize communication throughput under solar EH, assuming a fixed computation time. This formulation constitutes a critical subproblem of the joint optimization framework presented in Section IV.

## A. Problem Formulation

The communication throughput maximization problem is formulated in (10), where we fix n and thus $t _ { 2 }$ is also fixed.

$$
\left( \mathrm { P 1 } \right) \operatorname* { m a x } _ { t _ { 1 } , P _ { \mathrm { c o m m } } ( \cdot ) } \ \int _ { a } ^ { b } r ( \tau ) \mathrm { d } \tau\tag{10a}
$$

$$
\mathrm { s . t . } P _ { \mathrm { c o m p } } ( t - t _ { \mathrm { r e q } } - t _ { 1 } ) \leq E ( t ) , \forall t \in [ t _ { \mathrm { r e q } } + t _ { 1 } , a ] ,\tag{10b}
$$

$$
\int _ { a } ^ { t } P _ { \mathrm { c o m m } } ( \tau ) \mathrm { d } \tau \leq E ( t ) - P _ { \mathrm { c o m p } } t _ { 2 } , \ \forall t \in [ a , b ] ,\tag{10c}
$$

where $r ( t )$ is given in (7). Since the waiting time $t _ { 1 }$ is continuous and the transmit power $P _ { \mathrm { c o m m } } ( \cdot )$ is a continuoustime function, directly solving problem (P1) is highly complex. A possible approach is to discretize the feasible range of $t _ { 1 }$ and, for each discretized value, apply the continuous-EH transmission result in [45]. However, this discretization generally yields only an approximate solution and incurs high computational complexity when a fine time resolution is required. Therefore, we first investigate the structural properties of the optimal policy in the following subsection.

## B. Properties of Optimal Solution

To derive the exact solution of (P1), we first analyze the properties of the optimal solution. The property of the optimal waiting time is characterized in the following proposition.

Proposition 1 (Optimal Waiting Time). To achieve the optimal solution, the waiting time $t _ { 1 }$ should be minimized, provided that the energy causality constraint for computation (10b) is satisfied. Specifically, the optimal value $t _ { 1 } ^ { * }$ is given by

$$
\begin{array} { r l } & { t _ { 1 } ^ { * } = \operatorname* { i n f } \Big \{ t _ { 1 } \in [ 0 , T - t _ { 2 } ] \Big | } \\ & { \quad P _ { \mathrm { c o m p } } ( t - t _ { \mathrm { r e q } } - t _ { 1 } ) \leq E ( t ) , \forall t \in [ t _ { \mathrm { r e q } } + t _ { 1 } , a ] \Big \} . } \end{array}\tag{11}
$$

Proof. Let $t _ { 1 } ^ { * }$ denote the minimal feasible waiting time defined in (11). We show that any $\hat { t } _ { 1 } ~ > ~ t _ { 1 } ^ { * }$ cannot yield a higher throughput than $t _ { 1 } ^ { * } .$ . Consider a feasible policy $( \hat { t } _ { 1 } , \hat { P } _ { \mathrm { c o m m } } ( \cdot ) )$ achieving a throughput $\hat { R } .$ Since $t _ { 1 } ^ { * } < \hat { t } _ { 1 }$ , the computation phase completes earlier, expanding the feasible time window for communication. Let $\begin{array} { r c l } { a ^ { * } } & { = } & { t _ { \mathrm { r e q } } + t _ { 1 } ^ { * } + t _ { 2 } } \end{array}$ and $\hat { a } \_ =$ $t _ { \mathrm { r e q } } + \hat { t } _ { 1 } + t _ { 2 }$ . By extending $\hat { P } _ { \mathrm { c o m m } } ( t )$ as zero over $[ a ^ { * } , \hat { a } )$ the resulting policy remains feasible for (P1) and achieves the same throughput ${ \hat { R } } .$ Since reducing $t _ { 1 }$ does not shrink the feasible set of communication policies, $t _ { 1 } ^ { * }$ is optimal. □

Remark 2 (Separation Principle). Proposition 1 reveals that the optimization of $\cdot _ { t _ { 1 } }$ and $P _ { \mathrm { c o m m } } ( \cdot )$ can be decoupled without compromising optimality. This separation significantly reduces the problem complexity, as it allows the optimal waiting time to be determined prior to solving for the power control.

Having determined the optimal waiting time $t _ { 1 } ^ { * }$ we proceed to optimize the communication policy. Let [a, b] denote the communication interval corresponding to $t _ { 1 } ^ { * }$ . We define the feasible set for the transmit power $P _ { \mathrm { c o m m } } ( \cdot )$ as

$$
\begin{array} { r l } & { \mathcal { B } \triangleq \Big \{ P _ { \mathrm { c o m m } } : [ a , b ]  [ 0 , \infty ) \Big | } \\ & { \qquad \displaystyle \int _ { a } ^ { t } P _ { \mathrm { c o m m } } ( \tau ) \mathrm { d } \tau \leq \bar { E } ( t ) , \quad \forall t \in [ a , b ] \Big \} , } \end{array}\tag{12}
$$

where $\bar { E } ( t ) \triangleq E ( t ) - P _ { \mathrm { c o m p } } t _ { 2 }$ represents the residual harvested energy available for communication. Since the task execution window $T$ is relatively small, the channel gain over the window can be assumed to be unchanged, i.e., $g ( t ) \equiv g$ . For simplicity, we let $\begin{array} { r } { \alpha \triangleq \frac { g } { N _ { 0 } B } , \phi ( P ) \triangleq B \log _ { 2 } ( 1 + \alpha P ) } \end{array}$ . Note that a is a linear function of $t _ { 1 }$ . Consequently, (P1) reduces to the following problem:

$$
\operatorname* { m a x } _ { P _ { \mathrm { c o m m } } ( \cdot ) \in \mathcal { B } } \quad \int _ { a } ^ { b } \phi \big ( P _ { \mathrm { c o m m } } ( \tau ) \big ) \mathrm { d } \tau .\tag{13}
$$

Since $E _ { \mathrm { c o m m } } ( t )$ and $P _ { \mathrm { c o m m } } ( t )$ have a one-to-one mapping, we use them interchangeably in the analysis. The property of optimal transmit power is given by the following theorem.

Theorem 1 (Optimal Policy via Lower Convex Envelope [45]). The optimal cumulative energy consumption $E _ { \mathrm { c o m m } } ^ { * } ( t ) f o r \ t \in$ $[ a , b ]$ is given by the lower convex envelope (LCE) of E¯(t), anchored at the initial point $( a , 0 )$ and the final point $\left( b , { \bar { E } } ( b ) \right)$

## C. Optimal Waiting Time and Transmit Power

Based on the preceding results, this subsection introduces the optimal solutions for (P1) under the representative solar-EH profile in (3). According to Proposition 1 and Theorem 1, a minimal possible waiting time and an LCE of the solar-EH profile are required. Leveraging the closed-form expression of the EH curve, we can determine a closed-form solution efficiently. For analytical clarity, we assume that $\begin{array} { r } { P _ { \mathrm { c o m p } } t _ { 2 } < \frac { P _ { 0 } } { \omega } } \end{array}$ $\begin{array} { r } { t _ { 2 } < T < \frac { \pi } { 2 \omega } } \end{array}$ , and $E _ { 0 } = 0$ . Other cases are either infeasible or can be handled using the same procedures. To provide a unified expression for transmit power when $\begin{array} { r } { 0 \leq t _ { \mathrm { r e q } } \leq \frac { 2 \pi } { \omega } - T } \end{array}$ , we first investigate two specific cases for $t _ { \mathrm { r e q } }$ as follows.

1) Case 1: Task Starts from Orbital Noon: We consider the scenario when $t _ { \mathrm { r e q } } = 0$ and derive the exact solution to (P1). We start with finding $t _ { 1 } ^ { * }$ . Since $E ( t )$ is concave over $[ 0 , \frac { \pi } { 2 \omega } ]$ tangency between $E ( t )$ and the computation-related segment is impossible. Therefore, the minimal feasible $t _ { 1 }$ occurs in either case as follows: (1) Immediate start. It is feasible if and only if $\textstyle { \frac { P _ { 0 } } { \omega } } \sin ( \omega t _ { 2 } ) \geq P _ { \mathrm { c o m p } } t _ { 2 }$ , and renders $t _ { 1 } ^ { * } ~ = ~ 0 . ( 2 )$ Terminal touch at $t _ { 1 } + t _ { 2 }$ . It is feasible if $\frac { P _ { 0 } } { \omega }$ sin $( \omega t _ { 2 } ) ~ < ~ P _ { \mathrm { c o m p } } t _ { 2 }$ . In this case, a positive wait is necessary, and t<sup>∗</sup> is characterized by tightness at the end of the computation window: $E ( t _ { 1 } ^ { * } +$ $\begin{array} { r } { \dot { t _ { 2 } } ) = \dot { P } _ { \mathrm { c o m p } } t _ { 2 } \Longleftrightarrow \sin \big ( \omega ( t _ { 1 } ^ { * } + t _ { 2 } ) \big ) = \frac { \omega P _ { \mathrm { c o m p } } } { P _ { 0 } } t _ { 2 } . } \end{array}$ , which yields

![](images/aa1bc19f4ebcdfe03f1ae1587b268ae0b2d8a3b3bd965f59f52adbc55a0a5919.jpg)  
(a) $t _ { \mathrm { r e q } } = 0 .$

![](images/05b1047cead58f5e9c8680bd90c857f1c9b67638409792ec709f66a2301f1d41.jpg)  
(b) $\begin{array} { r } { t _ { \mathrm { r e q } } = \frac { 3 \pi } { 2 \omega } . } \end{array}$  
Fig. 2: Energy evolution and time allocation under the optimal policy for two representative task-request times. Here, $E _ { c } ( t ) = E _ { \mathrm { c o m p } } ( t ) + \dot { E } _ { \mathrm { c o m m } } ( t )$ ) denotes the total cumulative energy consumption.

$$
t _ { 1 } ^ { * } ~ = ~ \frac { 1 } { \omega } \arcsin \left( \frac { \omega P _ { \mathrm { c o m p } } } { P _ { 0 } } t _ { 2 } \right) ~ - ~ t _ { 2 } ,\tag{14}
$$

well-defined when $\begin{array} { r } { P _ { \mathrm { c o m p } } t _ { 2 } < { \frac { P _ { 0 } } { \omega } } } \end{array}$ and $\begin{array} { r } { t _ { 1 } ^ { * } \in \left[ 0 , \frac { \pi } { 2 \omega } - t _ { 2 } \right] } \end{array}$ . In summary, the optimal $t _ { 1 } ^ { * }$ is given by

$$
\begin{array} { r } { t _ { 1 } ^ { * } = \left\{ \begin{array} { l l } { 0 , \qquad } & { \frac { P _ { 0 } } { \omega } \sin ( \omega t _ { 2 } ) \geq P _ { \mathrm { c o m p } } t _ { 2 } , } \\ { \frac { 1 } { \omega } \arcsin ( \frac { \omega P _ { \mathrm { c o m p } } } { P _ { 0 } } t _ { 2 } ) - t _ { 2 } , \frac { P _ { 0 } } { \omega } \sin ( \omega t _ { 2 } ) < P _ { \mathrm { c o m p } } t _ { 2 } . } \end{array} \right. } \end{array}\tag{15}
$$

For $P _ { \mathrm { c o m m } } ^ { * } ( t )$ , since $\bar { E } ( t )$ is concave on $[ a , b ] .$ , its LCE is the affine function interpolating (a, 0) and $\left( b , \bar { E } ( b ) \right)$ . Hence,

$$
E _ { \mathrm { c o m m } } ^ { * } ( t ) = \frac { t - a } { b - a } \bar { E } ( b ) = \frac { t - a } { b - a } \big [ E ( b ) - P _ { \mathrm { c o m p } } t _ { 2 } \big ] .\tag{16}
$$

Taking the derivative yields the constant optimal power:

$$
P _ { \mathrm { c o m m } } ^ { * } ( t ) \equiv \frac { E ( b ) - P _ { \mathrm { c o m p } } t _ { 2 } } { b - a } = \frac { \frac { P _ { 0 } } { \omega } \sin ( \omega b ) - P _ { \mathrm { c o m p } } t _ { 2 } } { b - a } .\tag{17}
$$

For some parameters, the optimal policy is shown in Fig. $2 ( \mathrm { a } ) . ^ { 2 }$

2) Case 2: Task Startsfrom Orbital Dawn: We consider the scenario when $\begin{array} { r } { t _ { \mathrm { r e q } } = \frac { 3 \pi } { 2 \omega } } \end{array}$ . Under the assumption $P _ { \mathrm { c o m p } } t _ { 2 } <$ $\frac { P _ { 0 } } { \omega }$ , the energy-causality constraint is satisfied even with $t _ { 1 } =$ 0. Hence $t _ { 1 } ^ { * } = 0$ . For $P _ { \mathrm { c o m m } } ^ { * } ( t )$ , we first investigate a tangent construction. Define the slope of the supporting line anchored

at (a, 0) as

$$
s \ \triangleq \ \operatorname* { i n f } _ { t \in ( a , b ] } \ \frac { \bar { E } ( t ) - 0 } { t - a } \ = \ \operatorname* { i n f } _ { t \in ( a , b ] } \ \frac { E ( t ) - P _ { \mathrm { c o m p } } t _ { 2 } } { t - a } .\tag{18}
$$

Since $\bar { E }$ is convex on $( a , b ]$ , the infimum is achieved at a unique point $\tau \in ( a , b ]$ . If no interior tangency exists, the minimizer occurs at the boundary, and we set $\tau \triangleq b .$ . When an interior tangency exists, τ satisfies the equal-slope condition $\frac { \bar { E } ( \tau ) - 0 } { \tau - a } ~ = ~ \bar { E } ^ { \prime } \bar { ( \tau ) }$ . With explicit form of $\bar { E } ( t )$ , it becomes

$$
\frac { \frac { P _ { 0 } } { \omega } \big [ 2 + \sin ( \omega \tau ) \big ] - P _ { \mathrm { c o m p } } t _ { 2 } } { \tau - a } ~ = ~ P _ { 0 } \cos ( \omega \tau ) .\tag{19}
$$

After transforming it to $P _ { 0 } \big [ 2 { + } \mathrm { s i n } ( \omega \tau ) \big ] - \omega P _ { \mathrm { c o m p } } t _ { 2 } { - } P _ { 0 } \omega ( \tau -$ $a ) \cos ( \omega \tau ) = 0$ , it follows that the left-hand side is monotone; the equation can therefore be solved efficiently by bisection. Therefore, $E _ { \mathrm { c o m m } } ^ { * } ( t )$ and the corresponding $P _ { \mathrm { c o m m } } ^ { * } ( t )$ are

$$
E _ { \mathrm { c o m m } } ^ { * } ( t ) ~ = ~ \left\{ \begin{array} { l l } { \displaystyle \frac { t - a } { \tau - a } \bar { E } ( \tau ) , } & { t \in [ a , \tau ] , } \\ { \bar { E } ( t ) , } & { t \in [ \tau , b ] , } \end{array} \right.\tag{20}
$$

$$
P _ { \mathrm { c o m m } } ^ { * } ( t ) = \left\{ \begin{array} { l l } { \displaystyle \frac { \bar { E } ( \tau ) } { \tau - a } = \frac { E ( \tau ) - P _ { \mathrm { c o m p } } t _ { 2 } } { \tau - a } , } & { t \in [ a , \tau ] , } \\ { \displaystyle \bar { E } ^ { \prime } ( t ) = P _ { 0 } \cos ( \omega t ) , } & { t \in ( \tau , b ] . } \end{array} \right.\tag{21}
$$

Note that when $t \in ( \tau , b ] , P _ { \mathrm { c o m m } } ^ { * } ( t )$ follows a cosine-power control. If no interior tangency exists $( \mathrm { i . e . , } \ \tau = b ) , P _ { \mathrm { c o m m } } ^ { * } ( t )$ reduces to a constant-power policy $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { P _ { \mathrm { c o m m } } ^ { * } ( t ) \equiv \frac { \bar { E } ( b ) } { b - a } } \end{array}$ . For some parameters, the optimal policy is shown in $\mathrm { F i g . } 2 ( \mathbf { b } )$

3) Optimal Solutions: Based on the previous discussions of the two special cases, we have identified the basic structure of the solutions, which naturally extends to the general case. We first examine $t _ { 1 } ^ { * }$ w.r.t. different $t _ { \mathrm { r e q } } .$ . From (14), define

$$
\tau _ { 0 } \triangleq \frac { 1 } { \omega } \arcsin \left( \frac { \omega P _ { \mathrm { c o m p } } } { P _ { 0 } } t _ { 2 } \right) \ - \ t _ { 2 } .\tag{22}
$$

The causality constraint requires $t _ { \mathrm { r e q } } + t _ { 1 } \geq \tau _ { 0 }$ . If $\tau _ { 0 } > 0$ , it implies $t _ { 1 } \geq \tau _ { 0 } - t _ { \mathrm { r e q } }$ . Otherwise, if $\tau _ { 0 } \leq 0 .$ , then $t _ { \mathrm { r e q } } + t _ { 1 } \geq \tau _ { 0 }$ always holds. Therefore, we have

$$
t _ { 1 } ^ { * } = \operatorname* { m a x } \{ 0 , \ \tau _ { 0 } - t _ { \mathrm { r e q } } \} .\tag{23}
$$

Next, we examine $P _ { \mathrm { c o m m } } ^ { * } ( t )$ . When $\begin{array} { r } { t _ { \mathrm { r e q } } \leq \frac { 3 \pi } { 2 \omega } - T } \end{array}$ , E(t) is concave over $[ a , b ]$ , constant power applies: $\begin{array} { r } { P _ { \mathrm { c o m m } } ^ { * } ( t ) \equiv \frac { \bar { E } ( b ) } { b - a } } \end{array}$ When $\begin{array} { r } { \frac { 3 \pi } { 2 \omega } - T \leq t _ { \mathrm { r e q } } \leq \frac { 2 \pi } { \omega } - T , } \end{array}$ , E(t) is convex over $[ a , b ]$ and the tangent construction (case 2) applies, yielding (24).

## D. Discussion

1) The Meaning $o f \tau _ { 0 } .$ : The quantity $\tau _ { 0 }$ is a virtual threshold time defined by the endpoint-tightness condition: $E ( \tau _ { 0 } + t _ { 2 } ) =$ $P _ { \mathrm { c o m p } } t _ { 2 }$ . It indicates the earliest computation-start threshold implied by the harvested-energy curve. If $\tau _ { 0 } ~ > ~ 0$ , starting before $\tau _ { 0 }$ violates the computation energy-causality constraint, and the satellite must wait until $\tau _ { 0 } . \mathrm { { I f } } \tau _ { 0 } \leq 0 .$ , this threshold lies before the considered task horizon, so immediate computation is already feasible. Thus, (23) holds.

2) Constant-Power Control Versus Cosine-Power Control: The optimal power structure is determined by the convexity

$$
\begin{array} { r } { P _ { \mathrm { c o m m } } ^ { * } ( t ) = \left\{ \begin{array} { l l } { \displaystyle \frac { E ( b ) - P _ { \mathrm { c o m p } } t _ { 2 } } { b - a } , } & { t \in [ a , b ] , } \\ { \displaystyle \left\{ \frac { E ( \tau ) - P _ { \mathrm { c o m p } } t _ { 2 } } { \tau - a } , \right. } & { t \in [ a , \tau ] , } \\ { \displaystyle P ( t ) , } & { t \in ( \tau , b ) , } \end{array} \right. \mathrm { i f } \ t _ { \mathrm { r e q } } \leq \frac { 3 \pi } { 2 \omega } - T , } \end{array}\tag{24}
$$

![](images/9a84543ce43c9874fa02d28dd151bc653b2bfe8751fb7e3047151da6b401f209.jpg)  
Fig. 3: CLIP score versus DDIM steps.

of $E ( t )$ over $[ a , b ]$ . If $E ( t )$ is concave, i.e., $\begin{array} { r } { t _ { \mathrm { r e q } } \leq \frac { 3 \pi } { 2 \omega } - T , } \end{array}$ , the LCE is the chord from $( a , 0 )$ to $\left( b , { \bar { E } } ( b ) \right)$ ), yielding constantpower control. If $E ( t )$ is convex, i.e., $\begin{array} { r } { \frac { 3 \pi } { 2 \omega } - T \leq t _ { \mathrm { r e q } } \leq \frac { 2 \pi } { \omega } - T , } \end{array}$ the policy is constant on $[ a , \tau ]$ and then follows cosinepower control, i.e., $P _ { \mathrm { c o m m } } ^ { * } ( t ) ~ = ~ P ( t ) ~ = ~ P _ { 0 } \cos ( \omega t )$ on $( \tau , b )$ . This cosine-power control is a distinctive feature of the considered solar-EH-driven space AI system, where the optimal transmit power can track the orbital solar-EH profile. Nevertheless, constant-power control applies in most cases, covering approximately three quarters of the task-start range.

3) Extension to a General Solar-EH Profile: Although the previous subsection evaluates a representative profile for demonstration, the properties established in Sec. III-B hold for any solar-EH profile. For example, for an arbitrary solar panel normal vector $\mathbf { n } = [ n _ { x } , n _ { y } , n _ { z } ] ^ { T }$ , the solar-EH profile is

$$
\begin{array} { r l } & { P ( t ) = \eta \cdot \gamma \cdot A \cdot \operatorname* { m a x } ( - n _ { x } \cos \beta \sin ( \omega t ) } \\ & { \qquad - n _ { y } \sin \beta - n _ { z } \cos \beta \cos ( \omega t ) , 0 ) . } \end{array}\tag{25}
$$

See Appendix A for the derivation. For a sanity check, substituting $\textbf { n } = \ [ 0 , 0 , - 1 ] ^ { T }$ into (25) recovers (2). The corresponding $E ( t )$ can be obtained by integrating $P ( t )$ . The solution derivations under such profiles are omitted for brevity.

## IV. JOINT $C ^ { 2 }$ UTILITY OPTIMIZATION UNDER SOLAR EH

In this section, we aim to optimize the energy and time resources between computation and communication $( \mathbf { C } ^ { 2 } )$ under solar EH.

## A. Problem Formulation

1) Metric Design: We consider two critical yet inherently conflicting objectives for space generative AI (see Section II-D): (onboard) CLIP score and communication throughput. To evaluate the overall performance, we define the joint $C ^ { 2 }$ utility as a composite metric of these two terms:

$$
U = S + \lambda R ,\tag{26}
$$

where λ is a balancing coefficient.

2) Approximation of $C L I P$ Score: As shown in Fig. 3, the CLIP score increases rapidly at early DDIM steps, while additional steps provide only marginal gains. Motivated by this saturation behavior, we restrict the DDIM step n to the effective set $\mathcal { N } \triangleq \{ n _ { \operatorname* { m i n } } , n _ { \operatorname* { m i n } } + 1 , \dots , n _ { \operatorname* { m a x } } \}$ , and approximate the onboard CLIP score within this set as

$$
S ( n ) \approx c _ { 3 } n + c _ { 4 } ,\tag{27}
$$

where $c _ { 3 } > 0$ and $c _ { 4 }$ are fitting coefficients.

3) Problem Description: We aim to maximize the joint $C ^ { 2 }$ utility by optimizing the DDIM steps, waiting time, and power control under the given solar-EH profile. A critical constraint is energy causality, ensuring that total energy consumption never exceeds the cumulative harvested energy. Let $a ( n ) \triangleq t _ { \mathrm { r e q } } + t _ { 1 } + t _ { 2 } ( n )$ denote the start of the communication phase. Accordingly, we formulate the joint optimization over the DDIM step $n \in \mathcal N$ , waiting time $t _ { 1 } .$ , and transmit-power policy $P _ { \mathrm { c o m m } } ( \cdot )$ as follows:

$$
( \mathrm { P 3 } ) \operatorname* { m a x } _ { \substack { n , t _ { 1 } , P _ { \mathrm { c o m m } } ( \cdot ) } } c _ { 3 } n + c _ { 4 } + \lambda \int _ { a ( n ) } ^ { b } \phi \big ( P _ { \mathrm { c o m m } } ( t ) \big ) \mathrm { d } t\tag{28a}
$$

$$
\mathrm { s . t . } P _ { \mathrm { c o m p } } ( t - t _ { \mathrm { r e q } } - t _ { 1 } ) \leq E ( t ) ,
$$

$$
\forall t \in [ t _ { \mathrm { r e q } } + t _ { 1 } , a ( n ) ] ,\tag{28b}
$$

(28c)

$$
\int _ { a ( n ) } ^ { t } P _ { \mathrm { c o m m } } ( \tau ) \mathrm { d } \tau \leq E ( t ) - P _ { \mathrm { c o m p } } t _ { 2 } ( n ) .\tag{28d}
$$

$$
\forall t \in [ a ( n ) , b ] .\tag{28e}
$$

## B. Problem Decomposition

The main difficulty of (P3) lies in the coupling among computation, communication, and solar EH. Computation and communication draw from the same solar-EH-powered energy buffer, creating a time-causal resource conflict between the two stages. To this end, we decompose (P3) with respect to the DDIM step and characterize the resulting optimality structure in the following proposition.

Proposition 2 (Decomposition of (P3)). For any fixed $n \in$ ${ \mathcal { N } } ,$ let $R ^ { * } ( n )$ denote the maximum throughput obtained by solving the communication-throughput maximization problem $( P l )$ with $t _ { 2 } = t _ { 2 } ( n )$ . Then, the optimal DDIM step of (P3) is given by

$$
n ^ { * } = \arg \operatorname* { m a x } _ { n \in N } \left\{ c _ { 3 } n + c _ { 4 } + \lambda R ^ { * } ( n ) \right\} .\tag{29}
$$

Proof. For each fixed $n ,$ (P3) reduces to (P1); maximizing $c _ { 3 } n + c _ { 4 } + \lambda R ^ { * } ( n )$ over $n \in \mathcal N$ yields the result. □

This decomposition reduces the original mixed discretecontinuous problem to an outer one-dimensional search over $n \in \mathcal N$ and an inner communication-throughput maximization problem. The resulting search over ${ \mathcal { N } } .$ , referred to as exhaustive search, yields an exact solution to (P3). More importantly, the decomposition separates the role of computation from the communication-control subproblem, thereby providing the basis for the closed-form analysis in the sequel.

## C. Closed-form Analysis

Although exhaustive search can solve (P3) exactly over the discrete set ${ \mathcal { N } } ,$ such a search-based solution provides limited insight into how the optimal DDIM step n depends on system parameters. It also requires repeatedly solving the inner communication problem for all feasible $n .$ To reveal the structure of the $C ^ { 2 }$ trade-off and enable lightweight onboard adaptation, we analyze the dominant constant-power regime, in which the reduced problem admits a closed-form characterization. Specifically, we consider the following problem:

$$
( \mathrm { P 4 } ) \operatorname* { m a x } _ { n \in \mathcal { N } } c _ { 3 } n + c _ { 4 } + \lambda R ( n ) ,\tag{30}
$$

where $R ( n ) = \left( b - a ( n ) \right) \phi \bigl ( P _ { \mathrm { c o m m } } ( n ) \bigr )$ . Here, $P _ { \mathrm { c o m m } } ( n ) \ =$ $\frac { E _ { b } - P _ { \mathrm { c o m p } } \dot { t } _ { 2 } \dot { ( n ) } } { b - a ( n ) }$ with $E _ { b } \triangleq E ( b )$ . For analytical tractability, we relax n from the discrete set $\mathcal { N }$ to the interval $[ n _ { \mathrm { m i n } } , n _ { \mathrm { m a x } } ] .$ and later recover the integer solution by evaluating the nearest feasible DDIM steps. We set $t _ { 1 } = 0$ and impose the feasibility conditions $E _ { \mathrm { c o m p } } ( a ( n _ { \mathrm { m a x } } ) ) ~ < ~ E ( a ( n _ { \mathrm { m a x } } ) )$ and $a ( n ) < b ,$ which ensure sufficient computation energy and a non-empty communication window, respectively.

We first propose the following proposition to describe the properties of $R ( n )$

Proposition 3 (Properties of $R ( n ) ) . \ R ( n )$ is concave and decreases monotonically w.r.t. n.

Proof. See Appendix B.

The monotonicity of $R ( n )$ reflects the fundamental resource trade-off: increasing n consumes more energy, thereby reducing the power available for transmission. Furthermore, the concavity of $R ( n )$ implies that the marginal degradation in throughput accelerates as energy becomes scarcer.

Corollary 1 (Concavity of $U ( n ) )$ . The utility function $U ( n )$ is concave w.r.t. n.

Proof. Since $S ^ { \prime \prime } ( n ) = 0$ and $\lambda > 0$

$$
U ^ { \prime \prime } ( n ) = \lambda R ^ { \prime \prime } ( n ) \leq 0 .
$$

Hence, $U ( n )$ is concave.

Fig. 4 illustrates the result of Corollary 1 and the fundamental $C ^ { 2 }$ trade-off. As the DDIM step n increases, $S ( n )$ improves due to greater computational investment. However, this consumes more energy and reduces the budget available for transmission, thereby degrading the throughput $R ( n )$ . As a result, $U ( n )$ exhibits unimodal behavior, increasing initially before declining. The maximizer therefore represents the optimal balance between computation and communication, and shifts as the weighting parameter λ varies. Corollary 1 enables the following closed-form characterization.

![](images/bb6e7c8474eb263b253b28b03f767c386149879e002489a7d83fd23d54f8c07f.jpg)  
Fig. 4: The $C ^ { 2 }$ trade-off. The normalized utility, $\tilde { U } ( n )$ , is plotted.

Theorem 2 (Closed-form Step Selection). In the dominant constant-power regime, under the feasibility conditions stated above and assuming that a real stationary point exists, equation $U ^ { \prime } ( n ) = 0$ admits the following solution:

$$
\begin{array} { r } { \tilde { n } = \left\{ \begin{array} { l l } { a _ { 1 } + a _ { 2 } \frac { \alpha \Delta _ { E } } { D } \frac { W _ { 0 } \left( - D e ^ { - C } \right) } { W _ { 0 } \left( - D e ^ { - C } \right) + 1 } , } & { i f \Delta _ { E } > 0 , } \\ { a _ { 1 } + a _ { 2 } \frac { \alpha \Delta _ { E } } { D } \frac { W _ { - 1 } \left( - D e ^ { - C } \right) } { W _ { - 1 } \left( - D e ^ { - C } \right) + 1 } , } & { i f \Delta _ { E } < 0 , } \end{array} \right. } \end{array}\tag{31}
$$

where $\begin{array} { r } { a _ { 1 } \ \triangleq \ \frac { T - t _ { 1 } - c _ { 2 } } { c _ { 1 } } , \ a _ { 2 } \ \triangleq \ \frac { 1 } { c _ { 1 } } , \ C \ \triangleq \ 1 + \ \frac { a _ { 2 } c _ { 3 } } { \lambda B } } \end{array}$ ln 2, $D \ { \stackrel { \triangle } { = } }$ $1 + \alpha P _ { \mathrm { c o m p } } ,$ , and $\bar { \Delta } _ { E } \triangleq E _ { b } - \bar { P _ { \mathrm { c o m p } } } ( T - t _ { 1 } )$ $W _ { k } ( \cdot )$ denotes the Lambert W function with branch k $I 4 6 J .$ The solution $n ^ { * }$ to problem $( P 4 )$ is given by: $n ^ { * } \ = \ \arg \operatorname* { m a x } _ { n \in { \mathcal S } } U ( n )$ where $\mathcal { \bar { S } } \triangleq \{ \lfloor \lceil \tilde { n } \rceil _ { n _ { \operatorname* { m i n } } } ^ { n _ { \operatorname* { m a x } } } \} , \ : \ : \lceil [ \tilde { n } ] _ { n _ { \operatorname* { m i n } } } ^ { n _ { \operatorname* { m a x } } } \rceil \}$ is the candidate set. Here, $[ x ] _ { a } ^ { b } \triangleq \operatorname* { m i n } ( \operatorname* { m a x } ( x , a ) , b )$ denotes the clipping operator. and $\lceil \cdot \rceil$ denote the floor and ceiling functions, respectively. The degenerate case $\Delta E = 0 ,$ , for which $U ( n )$ becomes affine, is omitted for brevity.

Proof. See Appendix C.

The condition $\Delta _ { E } \gtrless 0$ creates a physical boundary equivalent to $P _ { \mathrm { c o m m } } \gtrless P _ { \mathrm { c o m p } } ,$ which can be verified by

$$
\Delta _ { E } \gtrless 0 \iff E _ { b } \gtrless P _ { \mathrm { c o m p } } ( T - t _ { 1 } ) \iff P _ { \mathrm { c o m m } } \gtrless P _ { \mathrm { c o m p } } .
$$

This distinction is pivotal because the Lambert $W$ function branches, $W _ { 0 }$ and $W _ { - 1 } .$ , exhibit opposing monotonicity. Consequently, $n ^ { * }$ responds differently to system parameters in the communication- and computation-dominant power regimes.

Remark 3 (Complexity Analysis). The closed-form rule in Theorem 2 reduces online decision-making to evaluating at most two candidate integers around n˜. Therefore, its online complexity is $O ( 1 )$ . In contrast, the exact exhaustive-search solution requires evaluating all feasible DDIM steps in ${ \mathcal { N } } ,$ leading to a complexity of $O ( | \mathcal { N } | )$ .

## V. EXPERIMENTAL RESULTS

## A. Experimental Settings

1) System and communication settings: We investigate a space generative AI framework, and the relevant system parameters are summarized in Table I. The communication and energy-harvesting settings largely follow the specifications in [6] and [16], respectively.

TABLE I: System Parameters
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Orbital angular velocity</td><td> $\omega$ </td><td> $2 \pi / 5 4 0 0 ~ \mathrm { r a d / s }$ </td></tr><tr><td>Computation power</td><td> $P _ { \mathrm { c o m p } }$ </td><td> $3 0 0 ~ \mathrm { W }$ </td></tr><tr><td>Channel bandwidth</td><td> $B$ </td><td> $5 \times 1 0 ^ { 3 } ~ \mathrm { H z }$ </td></tr><tr><td>Noise power spectral density</td><td> $N _ { 0 }$ </td><td> $- 1 7 4 ~ \mathrm { d B m / H z }$ </td></tr><tr><td>Additional attenuation factor Transmit/receive antenna gain</td><td> $L _ { \mathrm { A L } }$   $G _ { \mathrm { t x } }$  and  $G _ { \mathrm { r x } }$ </td><td> $- 5 \ \mathrm { d B }$  53 dB and 0 dB</td></tr><tr><td>Carrier frequency</td><td> $f _ { c }$ </td><td> $1 2 ~ \mathrm { G H z }$ </td></tr><tr><td>Task execution window</td><td> $T$ </td><td> $3 \mathrm { ~ s ~ }$ </td></tr><tr><td>Energy conversion efficiency</td><td> $\eta$ </td><td> $0 . 1 9$ </td></tr><tr><td>Solar irradiance constant</td><td> $\gamma$ </td><td> $\mathrm { 1 3 5 3 ~ W / m ^ { 2 } }$ </td></tr><tr><td>Solar panel area</td><td> $A$ </td><td> $2 . 0 ~ \mathrm { m } ^ { 2 }$ </td></tr><tr><td>Weighting parameter</td><td> $\lambda$ </td><td> $4 . 3 8 6 9 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Minimum DDIM step</td><td> $n _ { \mathrm { m i n } }$ </td><td> $4$ </td></tr><tr><td>Maximum DDIM step</td><td> $n _ { \mathrm { m a x } }$ </td><td> $1 0$ </td></tr><tr><td>Initial energy at task arrival</td><td> $E _ { 0 }$ </td><td>400 J</td></tr></table>

2) Metrics: To evaluate the proposed framework, we adopt the following two metrics: (1) Communication Throughput: It is defined as the total number of bits delivered over the transmission window [a,b], and is used to quantify communication performance. (2) E2E CLIP Score: We measure the semantic similarity between the original text prompt and the image received at the ground device [38], to indicate the overall E2E generation quality. The reported metrics are averaged across diverse channel realizations and multiple queries.

3) Benchmarking schemes: We consider two categories of benchmarks. The first category is used to validate the proposed solver, while the second category evaluates the E2E performance gains of our joint $\bar { C ^ { 2 } }$ control.

• Solver benchmark: We compare the proposed closed-form solution with exhaustive search over the feasible set $\mathcal { N } = \{ n _ { \operatorname* { m i n } } , \dots , n _ { \operatorname* { m a x } } \}$ . It verifies that the closed-form solution guarantees optimality while avoiding the $O ( | \mathcal { N } | )$ search complexity.

• Computation-centric scheme: It enforces a fixed, high DDIM step $\begin{array} { r } { n \ = \ n _ { \operatorname* { m a x } } } \end{array}$ to prioritize image fidelity, potentially at the expense of transmission reliability in energy-scarce regimes.

• Communication-centric scheme: It prioritizes energy allocation for the downlink transmission to ensure high throughput, while only guaranteeing the minimal required DDIM step $n = n _ { \mathrm { m i n } }$

## B. Validation of Throughput Maximization (P1)

To isolate the gains derived from the optimal waiting time and transmit power, in this subsection, we fix the DDIM step to $n = 5$ and investigate the communication throughput. The following three baselines are considered: (1) Fixed waiting time and optimal transmit power. $t _ { 1 } = 0 . 4 ~ \mathrm { s }$ , with the optimal power obtained from Theorem 1. (2) Optimal waiting time and fixed transmit power. $t _ { 1 } ^ { * }$ is obtained from Proposition 1, with $P _ { \mathrm { c o m m } } = 1 0 ~ \mathrm { W } .$ (3) Fixed waiting time and fixed transmit power. Both the waiting time and transmit power are fixed: $t _ { 1 } = 0 . 4 ~ \mathrm { s }$ and $P _ { \mathrm { c o m m } } = 1 0 ~ \mathrm { W } .$

![](images/33b5b56d243ee998c10c13314a96c28e9051b2720da60f29baa3402c9ef3f3d9.jpg)

(a) Average communication throughput versus A.  
![](images/215deeffc0f354fe2d930880ffe7e25a0015f446ed3c819a5e5ef72325461f5e.jpg)  
(b) Average communication throughput versus $\beta .$  
Fig. 5: Communication throughput across different schemes.

As illustrated in Fig. 5, our proposed strategy consistently achieves the highest communication throughput across the entire range of panel areas A and solar elevation angles $\beta .$ In Fig. 5(a), the throughput grows logarithmically with A, demonstrating that our joint optimization of $t _ { 1 } ^ { * }$ and $P _ { \mathrm { c o m m } } ^ { * } ( t )$ effectively converts increased harvested energy into higher data volume. In Fig. 5(b), we observe that throughput remains stable at low elevation angles but degrades as $\beta$ approaches $9 0 °$ , reflecting the reduction in the effective solar harvesting area as the sun vector becomes orthogonal to the orbital plane. The persistent performance gap between our scheme and the baselines demonstrates the necessity of minimizing the waiting phase and adopting the LCE principle.

## C. Evaluation of the Closed-form Solution for (P4)

As shown in Fig. 6, the pre-rounding closed-form solution closely tracks the exhaustive-search optimum across the considered settings. This demonstrates that the closed-form rule maintains solution accuracy while reducing complexity. Analyzing the communication-dominant scenario reveals key resource trade-offs. First, $n ^ { * }$ decreases as the throughput weight λ increases (Fig. 6(a)), shifting resources toward transmission. A similar trend is observed for the bandwidth B (Fig. 6(c)). Conversely, higher initial energy $E _ { 0 }$ allows for more computation, increasing $n ^ { * }$ (Fig. 6(b)). Interestingly, $n ^ { * }$ exhibits a non-monotonic relationship with the deadline $T$ (Fig. 6(d)). Under tight deadlines $( T < 1 . 5 ~ \mathrm { s } )$ , transmission is severely bottlenecked, so the system prioritizes computation to preserve generation quality. As $T$ relaxes further, $n ^ { * }$ steadily decreases to fully exploit the extended window for throughput.

![](images/3ecd9516293dd8e47bd771dca596cae2a0ba6ff73b32595ba5ce2602b6daa839.jpg)  
(a) $n ^ { * }$ versus $\lambda .$

![](images/97bcc4fb085ff33f98b9c133118538459e17e8d9e93442c04c6235730749a684.jpg)

![](images/e8dfe33ef4bfa69cf329bab2e732c23b34d98edae91bd44712c0b335b88cfa91.jpg)  
(c) $n ^ { * }$ versus B.

(b) $n ^ { * }$ versus $E _ { 0 } .$  
![](images/1eb34622e918c5c81f568d4a705c5d20dafa780b59aca1609db803c7eb014d01.jpg)  
(d) $n ^ { * }$ versus $T .$

Fig. 6: Comparison of the optimal DDIM steps obtained by exhaustive search and the pre-rounding closed-form solution under different system parameters.  
![](images/bc8e4acdbff2a4228dc78c2546880f7567102d5ac497d219bec8c7a6d5761a38.jpg)

(a) Average E2E CLIP score versus $P _ { 0 } .$  
![](images/819ccad436c3ed0cda344dff8c79b0749fc3d4f5ed2f21656b9c5a3a647d2be4.jpg)  
(b) Average E2E CLIP score versus $E _ { 0 } .$  
Fig. 7: Performance comparison of various schemes under solar EH.

## D. Performance Evaluation

In this subsection, we evaluate the holistic performance of our joint $C ^ { 2 }$ optimization scheme.

1) Comparison with Other Schemes: As shown in Fig. 7, the proposed joint $C ^ { 2 }$ policy achieves the highest E2E CLIP score across the considered energy range. In the lowenergy regime, it reduces the DDIM step and approaches the communication-centric baseline, preserving sufficient energy for transmission. In the high-energy regime, it allocates more resources to generation and approaches the computationcentric baseline. The largest gain appears in the moderateenergy regime, where neither extreme allocation is optimal. The proposed policy effectively balances onboard generation quality and downlink throughput. The sharp transition of each scheme occurs when the available energy crosses its schemedependent feasibility threshold for completing both generation and transmission. Below the corresponding threshold, insufficient energy remains for downlink transmission after generation, resulting in transmission failure.

![](images/d52ba73e809632c87d20a59f192f7469c06e37bf2c3e97f330354fa2ba8f1cb0.jpg)

(a) Average communication throughput.  
![](images/2aaabfb4e2c33444f4daefc56a94daf2fe611f9f966da193b37448ba91fd91e4.jpg)  
(b) Average E2E CLIP score.  
Fig. 8: System performance at different orbital angles $( \theta \in [ 0 ^ { \circ } , 3 6 0 ^ { \circ } ] )$

2) Impact of Orbital Position: Fig. 8 illustrates the communication throughput and E2E CLIP score for independent task executions initiated at different orbital angles, each with initial energy $E _ { 0 }$ and the corresponding solar-EH profile. Although the communication-centric baseline achieves the highest throughput by minimizing the computation load, its E2E CLIP score remains lower than the other schemes in sunlit regions because the onboard generation quality is limited by the minimum DDIM step. In contrast, the computation-centric baseline achieves a high E2E CLIP score in sunlit regions, but suffers a sharp performance drop during the eclipse. This is because the fixed high DDIM step consumes excessive computation energy when no solar energy is harvested, leaving insufficient energy for downlink transmission. As a result, transmission fails and the E2E CLIP score drops to zero. The proposed joint $C ^ { 2 }$ policy avoids both drawbacks by adapting the DDIM step to the orbital energy state: it allocates more resources to generation when sunlight is available and scales back computation during the eclipse to preserve downlink transmission. Consequently, it maintains the highest E2E CLIP score over the full orbit.

3) Impact of Solar EH: Fig. 9 shows the impact of solar EH under different $E _ { 0 }$ . In the resource-constrained regime, solar EH substantially improves the E2E CLIP score because the harvested energy helps satisfy the minimum generationand-transmission requirement and can further support a larger DDIM step or higher downlink throughput. The visible steplike transitions are caused by discrete DDIM step selection:

![](images/adf78ba1cbfd470230512eea946defd8f10b75bed67bf7568f3b8316febf5584.jpg)  
Fig. 9: Impact of solar EH on the average E2E CLIP score.

![](images/b613e85ac9fadfdd70cb34372e7765670b706eaf8e53e2961ea11c4d685d61e1.jpg)  
Fig. 10: Comparison of the average E2E CLIP score of different schemes in a specific scenario, evaluated with and without solar EH.

once the available energy exceeds a threshold that supports an additional DDIM step while preserving reliable transmission, n increases, resulting in a sudden improvement in the E2E CLIP score. As $E _ { 0 }$ becomes sufficiently large, the curves with and without solar EH converge because the initial energy alone is enough to support both generation and transmission, making additional harvested energy less influential. Fig. 10 further compares different schemes with and without solar EH in a representative scenario. With solar EH, the proposed scheme achieves the highest E2E CLIP score by jointly adapting computation and communication. Without solar EH, the computation-centric baseline becomes transmission-infeasible because its fixed high DDIM step depletes the energy budget before downlink transmission, causing transmission failure. The communication-centric baseline remains stable but quality-limited because it always prioritizes transmission and uses the minimum DDIM step. In contrast, the proposed scheme scales back computation when solar EH is unavailable, preserving transmission reliability while maintaining a competitive E2E CLIP score. The qualitative examples in Fig. 11 are consistent with these results: solar EH improves visual quality, and without solar EH, the proposed scheme still delivers semantically aligned images while the computationcentric baseline fails to transmit.

## VI. CONCLUDING REMARKS

This work establishes a framework for space generative AI and reveals the inherent coupled $C ^ { 2 }$ bottleneck under solar EH. Specifically, it characterizes how a satellite should jointly manage waiting, diffusion-based generation, and downlink transmission when $C ^ { 2 }$ stages draw from the same time-causal harvested-energy buffer. First, computation should start at the earliest energy-feasible time. Second, constant transmit power is optimal in most orbital states, while dawn may require solar-tracking control. Third, adaptive generation depth is most beneficial under moderate energy, where neither computationnor communication-centric extremes are optimal. These results advance satellite edge intelligence from generic computationoffloading design towards task-oriented generative service provisioning under physical orbital constraints.

![](images/889d385c4fd913894633ad53ce6e96467c0e92753e0e73cd063a5ad4ce7a004c.jpg)

(a) Visual performance with solar EH.  
![](images/0291765ac8455a262d161dfe5d55d11789fb251e98ec425df466710fa9f745d4.jpg)  
(b) Visual performance without solar EH.  
Fig. 11: Qualitative comparison of generated images.

Future work should extend this framework to multi-user and multi-task satellite systems, where heterogeneous generative requests compete for shared harvested energy. It is also important to incorporate practical hardware constraints such as finite battery capacity, peak transmit power, and accelerator-dependent inference energy. Finally, more refined E2E quality models are needed to jointly capture generation quality, compression distortion, channel effects, and humanperceived semantic fidelity.

## APPENDIX

## A. Derivation for Eq. (25)

As the satellite orbits the Earth with angular velocity $\omega ,$ the $\mathcal { F } _ { b }$ rotates relative to the $\mathcal { F } _ { o }$ around the $Y _ { o } .$ . The rotation matrix from $\mathcal { F } _ { o }$ to $\mathcal { F } _ { b }$ at time t is:

$$
\mathbf { R } ( t ) = \left[ \begin{array} { c c c } { - \sin ( \omega t ) } & { 0 } & { - \cos ( \omega t ) } \\ { 0 } & { - 1 } & { 0 } \\ { - \cos ( \omega t ) } & { 0 } & { \sin ( \omega t ) } \end{array} \right] .\tag{32}
$$

The sun vector in $\mathcal { F } _ { b }$ is derived as $\mathbf S _ { b } ( t ) = \mathbf R ( t ) \mathbf S _ { o }$

$$
\begin{array} { r } { { \bf S } _ { b } ( t ) = \left[ { - \cos \beta \sin ( \omega t ) } { \imath } \right] . } \\ { - \sin \beta } \\ { - \cos \beta \cos ( \omega t ) } \end{array}\tag{33}
$$

The instantaneous harvested power is proportional to the effective projection area:

$$
P ( t ) = \eta \cdot \gamma \cdot A \cdot \operatorname* { m a x } \left( \mathbf { n } \cdot \mathbf { S } _ { b } ( t ) , 0 \right) .\tag{34}
$$

Taking the inner product yields the results.

## B. Proof of Proposition 3

For $R ( n )$ , with $u \triangleq T - t _ { 1 } - c _ { 1 } n - c _ { 2 }$ and $\Delta _ { E } \triangleq E _ { b } -$ $P _ { \mathrm { c o m p } } ( T - t _ { 1 } )$ , it becomes

$$
f ( u ) = B u \log _ { 2 } \Bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \frac { \alpha \Delta _ { E } } { u } \Bigr ) , \quad u > 0 .\tag{35}
$$

Calculating its first derivative yields

$$
\begin{array} { r l } & { f ^ { \prime } ( u ) = B \Bigg [ \log _ { 2 } \Bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \frac { \alpha \Delta _ { E } } { u } \Bigr ) } \\ &  \qquad + \cfrac { u ^ { \mathrm { L } } } { \ln 2 } \cfrac { d }  \cfrac { d u }  \cfrac { d u }  \cfrac { d u }  \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \coth } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \cfrac { d u } { \coth } { d } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } \end{array}\tag{36}
$$

The second derivative is

$$
\begin{array} { c } { { f ^ { \prime \prime } ( u ) = B \displaystyle \frac { d } { d u } \Biggl \{ \log _ { 2 } \Bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \frac { \alpha \Delta _ { E } } { u } \Bigr ) } } \\ { { - \ \displaystyle \frac { 1 } { \ln 2 } \frac { \alpha \Delta _ { E } } { u \bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u \bigr ) } \Biggr \} } } \\ { { = \displaystyle \frac { B } { \ln 2 } \Biggl [ \frac { - \alpha \Delta _ { E } / u ^ { 2 } } { 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u } } } \\ { { - \ \displaystyle \frac { d } { d u } \left( \frac { \alpha \Delta _ { E } } { u \bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u \bigr ) } \right) \Biggr ] . } } \end{array}\tag{37}
$$

Compute the remaining derivative. Let $\psi ( u ) \triangleq 1 + \alpha P _ { \mathrm { c o m p } } +$ $\alpha \Delta _ { E } / u$ and $\begin{array} { r } { q ( u ) \triangleq \frac { \bar { \alpha } \Delta _ { E } } { u \psi ( u ) } } \end{array}$ . Then $\psi ^ { \prime } ( u ) = - \alpha \Delta _ { E } / u ^ { 2 }$ , and

$$
\begin{array} { l } { { q ^ { \prime } ( u ) = \alpha \Delta _ { E } \cdot \frac { d } { d u } \big ( u ^ { - 1 } \psi ( u ) ^ { - 1 } \big ) } } \\ { { \ = \alpha \Delta _ { E } \left[ - u ^ { - 2 } \psi ^ { - 1 } + u ^ { - 1 } \cdot \left( - 1 \right) \psi ^ { - 2 } \psi ^ { \prime } ( u ) \right] } } \\ { { \ = - \ \frac { \alpha \Delta _ { E } } { u ^ { 2 } \psi ( u ) } \ + \ \frac { \alpha \Delta _ { E } } { u } \cdot \frac { \alpha \Delta _ { E } / u ^ { 2 } } { \psi ( u ) ^ { 2 } } } } \\ { { \ = - \ \frac { \alpha \Delta _ { E } } { u ^ { 2 } \psi ( u ) } \ + \ \frac { \alpha ^ { 2 } \Delta _ { E } ^ { 2 } } { u ^ { 3 } \psi ( u ) ^ { 2 } } . } } \end{array}\tag{38}
$$

Plugging back, we have

$$
\begin{array} { r l r } {  { f ^ { \prime \prime } ( u ) = \frac { B } { \ln 2 } [ - \frac { \alpha \Delta _ { E } } { u ^ { 2 } \psi ( u ) } - ( - \frac { \alpha \Delta _ { E } } { u ^ { 2 } \psi ( u ) } + \frac { \alpha ^ { 2 } \Delta _ { E } ^ { 2 } } { u ^ { 3 } \psi ( u ) ^ { 2 } } ) ] } } \\ & { } & { = \frac { B } { \ln 2 } [ - \frac { \alpha ^ { 2 } \Delta _ { E } ^ { 2 } } { u ^ { 3 } \psi ( u ) ^ { 2 } } ] } \\ & { } & { = - \frac { B \alpha ^ { 2 } \Delta _ { E } ^ { 2 } } { \ln 2 } \frac { 1 } { u ^ { 3 } ( 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u ) ^ { 2 } } \le 0 . } \end{array}\tag{39}
$$

Therefore, $f ^ { \prime } ( u ) \geq f ^ { \prime } ( + \infty ) > 0$ . Hence, we have $R ^ { \prime \prime } ( n ) \leq 0$ and $R ^ { \prime } ( n ) < 0$ , which completes the proof.

## C. Proof of Theorem 2

Let $u \triangleq T - t _ { 1 } - c _ { 1 } n - c _ { 2 }$ , and conversely, $n = a _ { 1 } - a _ { 2 } u$ where $\begin{array} { r } { a _ { 1 } \triangleq \frac { T - t _ { 1 } - c _ { 2 } } { c _ { 1 } } , a _ { 2 } \triangleq \frac { 1 } { c _ { 1 } } } \end{array}$ . This transforms $U ( n )$ to

$$
\hat { U } ( u ) = - a _ { 2 } c _ { 3 } u + a _ { 1 } c _ { 3 } + c _ { 4 } + \lambda f ( u ) ,\tag{40}
$$

where $f ( u )$ is from (35). Consequently, it suffices to solve $\hat { U } ^ { \prime } ( u ) \stackrel { \cdot } { = } 0$ for the stationary point u˜, from which n˜ is recovered. Using the expression for $f ^ { \prime } ( u )$ from (36), the derivative $\hat { U } ^ { \prime } ( u )$ is explicitly given by

$$
\begin{array} { l } { \displaystyle \hat { U } ^ { \prime } ( u ) = - a _ { 2 } c _ { 3 } + \lambda B \left[ \log _ { 2 } \left( 1 + \alpha P _ { \mathrm { c o m p } } + \frac { \alpha \Delta _ { E } } { u } \right) \right. } \\ { \displaystyle \left. - \frac { 1 } { \ln 2 } \frac { \alpha \Delta _ { E } } { u ( 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u ) } \right] . } \end{array}\tag{41}
$$

Setting $\hat { U } ^ { \prime } ( u ) = 0$ gives

$$
\ln \Bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + { \frac { \alpha \Delta _ { E } } { u } } \Bigr ) - { \frac { \alpha \Delta _ { E } } { u \bigl ( 1 + \alpha P _ { \mathrm { c o m p } } + \alpha \Delta _ { E } / u \bigr ) } } = { \frac { a _ { 2 } c _ { 3 } } { \lambda B } } \ln 2 .
$$

Let $\begin{array} { r } { D \triangleq { 1 + \alpha P _ { \mathrm { c o m p } } } , K \triangleq \frac { a _ { 2 } c _ { 3 } } { \lambda B } } \end{array}$ ln 2, and $C \triangleq 1 + K$ . Then

$$
\ln \Bigl ( D + { \frac { \alpha \Delta _ { E } } { u } } \Bigr ) \ - \ { \frac { \alpha \Delta _ { E } } { u \bigl ( D + \alpha \Delta _ { E } / u \bigr ) } } \ = \ K .\tag{42}
$$

Let $\begin{array} { r } { x \triangleq D + \frac { \alpha \Delta _ { E } } { u } } \end{array}$ . Thus $\begin{array} { r } { \frac { \alpha \Delta _ { E } } { u } = x - D } \end{array}$ and $\begin{array} { r } { u = \frac { \alpha \Delta _ { E } } { x - D } , \ x \neq } \end{array}$ D. This further simplifies the equation as

$$
\ln x \ - \ { \frac { \alpha \Delta _ { E } } { u x } } \ = \ K .
$$

Plugging $\begin{array} { r } { \frac { \alpha \Delta _ { E } } { u x } \ = \ \frac { x - D } { x } } \end{array}$ into it, we have

$$
\ln x \ + \ { \frac { D } { x } } \ = \ 1 + K \ = \ C .
$$

Transform it and we have $x = e ^ { C - D / x } = e ^ { C } e ^ { - D / x }$ . Then, $x e ^ { D / x } = e ^ { C }$ . Let $y \ \triangleq \ { \frac { D } { x } } , \ x \ = \ { \frac { D } { y } }$ , then $\begin{array} { r } { \frac { D } { u } e ^ { y } = e ^ { C } } \end{array}$ and thus $\left( - y \right) e ^ { - y } = - D e ^ { - C }$ . Hence $- y = W _ { k } \big ( - D e ^ { - C } \big )$ , and $\begin{array} { r } { x = \frac { \ d _ { D } } { \ d u } = - D / W _ { k } \ d \mathopen { } \mathclose \bgroup \left( - D e ^ { - C } \aftergroup \egroup \right) } \end{array}$ with branch index k chosen to satisfy domain constraints. For brevity, we denote $\zeta _ { k } \equiv$ $W _ { k } \big ( - D e ^ { - C } \big )$ , and then $\begin{array} { r } { u \ = \ \frac { \alpha \Delta _ { E } } { - \frac { D } { \hat { \ r } _ { * } } - D } \ = \ - \frac { \alpha \Delta _ { E } } { D } \ \frac { \zeta _ { k } } { \zeta _ { k } + 1 } } \end{array}$ . We further investigate the term $\begin{array} { r } { y = \frac { \vec { D } ^ { \kappa } } { x } = \frac { 1 } { 1 + \frac { \alpha \Delta _ { E } } { D \kappa } } } \end{array}$ . All constituent parameters are positive, with the exception of $\Delta _ { E } .$ . If $\Delta _ { E } > 0$ it follows that $y < 1$ (implying $- y > - 1 )$ ; consequently, the solution corresponds to the principal branch $W _ { 0 }$ . If $\Delta _ { E } < 0$ the branch $W _ { - 1 }$ is selected. Therefore,

$$
\tilde { u } = \left\{ \begin{array} { l l } { - \frac { \alpha \Delta _ { E } } { D } \frac { W _ { 0 } \left( - D e ^ { - C } \right) } { W _ { 0 } \left( - D e ^ { - C } \right) + 1 } , } & { \mathrm { i f ~ } \Delta _ { E } > 0 , } \\ { - \frac { \alpha \Delta _ { E } } { D } \frac { W _ { - 1 } \left( - D e ^ { - C } \right) } { W _ { - 1 } \left( - D e ^ { - C } \right) + 1 } , } & { \mathrm { i f ~ } \Delta _ { E } < 0 . } \end{array} \right.\tag{43}
$$

Substituting (43) into $\tilde { n } = a _ { 1 } - a _ { 2 } \tilde { u }$ yields the results in (31). Due to the concavity of the objective function, the optimal integer solution $n ^ { * }$ is guaranteed to lie in the candidate set defined in Theorem 2. This completes the proof.

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 6840–6851, 2020.

[2] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., “Chain-of-thought prompting elicits reasoning in large language models,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 24824– 24837, 2022.

[3] G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, “Mobile edge intelligence for large language models: A contemporary survey,” IEEE Commun. Surveys Tuts., vol. 27, no. 6, pp. 3820–3860, 2025.

[4] Z. Wang, Q. Zeng, H. Zheng, and K. Huang, “Revisiting outage for edge inference systems,” IEEE Trans. Commun., 2026.

[5] Q. Chen, Z. Guo, W. Meng, S. Han, C. Li, and T. Q. Quek, “A survey on resource management in joint communication and computing-embedded sagin,” IEEE Commun. Surveys Tuts., vol. 27, no. 3, pp. 1911–1954, 2024.

[6] Q. Chen, X. Chen, and K. Huang, “Fedmeld: A model-dispersal federated learning framework for space-ground integrated networks,” arXiv preprint arXiv:2412.17231, 2024.

[7] H. H. Esmat, B. Lorenzo, and W. Shi, “Toward resilient network slicing for satellite–terrestrial edge computing iot,” IEEE Internet Things J., vol. 10, no. 16, pp. 14621–14645, 2023.

[8] Q. Chen, Z. Wang, X. Chen, J. Wen, D. Zhou, S. Ji, M. Sheng, and K. Huang, “Space–ground fluid ai for 6g edge intelligence,” Engineering, 2025.

[9] S. Ji, D. Zhou, M. Sheng, J. Li, and Z. Han, “Dynamic space-ground integrated mobility management strategy for mega leo satellite constellations,” IEEE Trans. Wireless Commun., vol. 23, no. 9, pp. 11043–11060, 2024.

[10] M. Sun, Z. Chen, J. Hou, K. Wang, and X. Chu, “Toward communication-efficient space data centers: Bottlenecks, architectures, and new paradigms,” arXiv preprint arXiv:2605.12681, 2026.

[11] Y. Xiao and X. Xu, “Sgicpnom: A computation offloading mechanism for 6g space-ground integrated computing power network,” Computer Networks, p. 112082, 2026.

[13] Starcloud, “Starcloud-1.” https://www.starcloud.com/starcloud-1, 2025.

[12] Z. Wang, H. Yang, M. Sheng, K. B. Letaief, and K. Huang, “Spacemoe: Realizing distributed mixture-of-experts inference over space networks,” arXiv preprint arXiv:2605.00515, 2026.

[14] S. Samsi, D. Zhao, J. McDonald, B. Li, A. Michaleas, M. Jones, W. Bergeron, J. Kepner, D. Tiwari, and V. Gadepally, “From words to watts: Benchmarking the energy costs of large language model inference,” in 2023 IEEE high performance extreme computing conference (HPEC), pp. 1–9, IEEE, 2023.

[15] M.-L. Ku, W. Li, Y. Chen, and K. R. Liu, “Advances in energy harvesting communications: Past, present, and future challenges,” IEEE Commun. Surveys Tuts., vol. 18, no. 2, pp. 1384–1412, 2015.

[16] Y. Yang, M. Xu, D. Wang, and Y. Wang, “Towards energy-efficient routing in satellite networks,” IEEE J. Sel. Areas Commun., vol. 34, no. 12, pp. 3869–3886, 2016.

[17] W. Liu, Z. Lai, Q. Wu, H. Li, Q. Zhang, Z. Li, Y. Li, and J. Liu, “In-orbit processing or not? sunlight-aware task scheduling for energyefficient space edge computing networks,” in IEEE INFOCOM 2024- IEEE Conference on Computer Communications, pp. 881–890, IEEE, 2024.

[18] Q. Li, S. Wang, X. Ma, A. Zhou, Y. Wang, G. Huang, and X. Liu, “Battery-aware energy optimization for satellite edge computing,” IEEE Transactions on Services Computing, vol. 17, no. 2, pp. 437–451, 2024.

[19] G. Qu, Z. Lin, Q. Chen, J. Li, F. Liu, X. Chen, and K. Huang, “Trimcaching: Parameter-sharing edge caching for ai model downloading,” IEEE Transactions on Networking, 2026.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

[21] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[22] O. Ozel, K. Tutuncuoglu, J. Yang, S. Ulukus, and A. Yener, “Transmission with energy harvesting nodes in fading wireless channels: Optimal policies,” IEEE J. Sel. Areas Commun., vol. 29, no. 8, pp. 1732–1743, 2011.

[23] K. Tutuncuoglu and A. Yener, “Optimum transmission policies for battery limited energy harvesting nodes,” IEEE Trans. Wireless Commun., vol. 11, no. 3, pp. 1180–1189, 2012.

[24] J. Yang and S. Ulukus, “Optimal packet scheduling in an energy harvesting communication system,” IEEE Trans. Commun., vol. 60, no. 1, pp. 220–230, 2011.

[25] S. Luo, R. Zhang, and T. J. Lim, “Optimal save-then-transmit protocol for energy harvesting wireless transmitters,” IEEE Trans. Wireless Commun., vol. 12, no. 3, pp. 1196–1207, 2013.

[26] D. Wen, S. Xie, X. Cao, Y. Cui, J. Xu, Y. Shi, and S. Cui, “Integrated sensing, communication, and computation for over-the-air federated edge learning,” IEEE Trans. Wireless Commun., vol. 25, pp. 2748–2762, 2026.

[27] Q. Ouyang, N. Ye, J. Gao, A. Wang, and L. Zhao, “Joint in-orbit computation and communication for minimizing download time from leo satellites,” IEEE Trans. Mob. Comput., vol. 23, no. 5, pp. 3950– 3963, 2023.

[28] K. Li, J. Jiao, J. Huang, Z. Xu, Q. Sun, X. Xu, Y. Wang, and Q. Zhang, “Age-critical joint communication and computation offloading for satellite-integrated internet,” IEEE Trans. Cogn. Commun. Netw., vol. 12, pp. 4387–4403, 2025.

[29] J. Cao, S. Zhang, Q. Chen, H. Wang, M. Wang, and N. Liu, “Computingaware routing for leo satellite networks: A transmission and computation integration approach,” IEEE Trans. Veh. Technol., vol. 72, no. 12, pp. 16607–16623, 2023.

[30] C. Ding, J.-B. Wang, H. Zhang, M. Lin, and G. Y. Li, “Joint optimization of transmission and computation resources for satellite and high altitude platform assisted edge computing,” IEEE Trans. Wireless Commun., vol. 21, no. 2, pp. 1362–1377, 2021.

[31] Q. Wang, X. Chen, and Q. Qi, “Energy-efficient design of satelliteterrestrial computing in 6g wireless networks,” IEEE Trans. Commun., vol. 72, no. 3, pp. 1759–1772, 2023.

[32] M. Dai, S. Chang, Y. Wang, and Z. Su, “Energy-efficient multi-access edge computing for heterogeneous satellite-maritime networks: A hybrid harvesting-and-offloading design,” IEEE Trans. Mob. Comput., 2025.

[33] L. P. Qian, X. Fan, M. Li, and Y. Wu, “Energy-efficient data gathering and computing in leo satellite-assisted marine iot networks,” IEEE Trans. Cogn. Commun. Netw., 2025.

[34] L. He, S. Li, Z. Jia, J. Wang, and Z. Han, “Joint data compression and task scheduling for leo satellite networks,” IEEE Trans. Veh. Technol., 2025.

[35] J. Lai, H. Liu, G. Xu, W. Jiang, X. Wang, and D. Jiang, “Joint computation offloading and resource allocation for leo satellite networks using hierarchical multi-agent reinforcement learning,” IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 4, pp. 2554–2567, 2024.

[36] Y. Gong, H. Yao, Z. Xiong, S. Guo, F. R. Yu, and D. Niyato, “Computation offloading and energy harvesting schemes for sum rate maximization in space-air-ground networks,” in GLOBECOM 2022- 2022 IEEE Global Communications Conference, pp. 3941–3946, IEEE, 2022.

[37] L. Zhang, J. Liu, M. Sheng, N. Zhao, and J. Li, “Exploiting collaborative computing to improve downlink sum rate in satellite integrated terrestrial networks,” IEEE Trans. Veh. Technol., vol. 72, no. 4, pp. 4670–4682, 2022.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning, pp. 8748–8763, PmLR, 2021.

[39] H. Cai, M. Li, Q. Zhang, M.-Y. Liu, and S. Han, “Condition-aware neural network for controlled image generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7194–7203, 2024.

[40] J. R. Wertz, Spacecraft attitude determination and control. Springer Science & Business Media, 2012.

[41] M. Cheriet, A. Bellar, M. Y. Ghaffour, A. Adnane, and M. A. Mohammed, “Inertia tensor estimation for a rigid nadir pointing satellite based on star tracker,” Advances in aircraft and spacecraft science, vol. 8, no. 2, pp. 111–126, 2021.

[42] R. R. Bate, D. D. Mueller, J. E. White, and W. W. Saylor, Fundamentals of astrodynamics. Courier Dover Publications, 2020.

[43] C. Cai, Y. Zhu, M. Sheng, J. Li, Y. Shi, D. Zhou, Z. Xie, and C. Zhang, “3c resources joint allocation for time-deterministic remote sensing image backhaul in the space-ground integrated network,” arXiv preprint arXiv:2510.09409, 2025.

[44] C. E. Shannon, “A mathematical theory of communication,” The Bell system technical journal, vol. 27, no. 3, pp. 379–423, 1948.

[45] B. Varan, K. Tutuncuoglu, and A. Yener, “Energy harvesting communications with continuous energy arrivals,” in 2014 Information Theory and Applications Workshop (ITA), pp. 1–10, IEEE, 2014.

[46] R. M. Corless, G. H. Gonnet, D. E. Hare, D. J. Jeffrey, and D. E. Knuth, “On the lambert w function,” Advances in Computational mathematics, vol. 5, no. 1, pp. 329–359, 1996.