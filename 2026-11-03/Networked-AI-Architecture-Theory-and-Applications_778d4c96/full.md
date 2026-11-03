# Networked AI: Architecture, Theory, and Applications

Liang Song , Jiayue Jin , Peng Sun ,Senior Member, IEEE, Di Li , Yang Liu ,Member, IEEE, Athanassios Skodras ,Life Senior Member, IEEE, Dimitris Ampeliotis ,Member, IEEE, Yong Yuan, Amit Dvir , Jiangchuan Liu ,Fellow, IEEE, Victor C. M. Leung , Life Fellow, IEEE, and Qi Bi,Fellow, IEEE

Abstract—Recent advances in Artificial Intelligence (AI), particularly large-scale deep learning, have been driven by monolithic or offline training on massive curated datasets. While successful, these approaches face structural limitations: escalating data and computational demands, limited online adaptability, and a lack of system-level intelligence in open, non-stationary environments. This paper presents Networked AI — a networked intelligent system composed of interacting, heterogeneous, and continuously evolving AI agents. We establish a hierarchical layered architecture comprising an Agent Layer for heterogeneous capability construction, an Interaction Layer for signalprocessing-guided online data extraction, and a Regulation Layer for stability-constrained collective evolution. Foundations in data science, signal processing, and system control are hereby synthesized into a unified theoretical framework of Networked AI, providing superior data efficiency, intelligence efficiency, interpretability, and scalability. Theoretical and experimental analysis demonstrates that Networked AI can approach nearlinear scaling in resource demand, in contrast to the exponential increase in standalone end-to-end systems. It provides not only online adaptation to out-of-distribution environmental novelties, but also the autonomous evolution towards application-specific intelligence. Application frontiers are discussed for IoT spaces, embodied, spatial, and industrial intelligences, as well as the cellular infrastructure and beyond. Key open challenges are identified towards realizing locally autonomous, collectively intelligent, continuously evolving, globally regulated Networked AI.

Index Terms—Networking Systems of AI, Deep Learning, Online Evolutive Learning, Multi-Agent Interaction, System Regulation, Cross-layer Resilience.

## I. INTRODUCTION

In the past decade, with the introduction of key technologies such as AlexNet [1], ResNet [2], Transformer [3], etc., Artificial Intelligence (AI) has entered a new era driven by foundation models, large-scale self-supervised learning, and transformer-based architectures. Recent advances in large language models (LLMs) [4], multimodal systems [5], and generative AI [6] have demonstrated unprecedented capabilities in language understanding, reasoning, perception, and content generation. At the same time, the rapidly growing applications of industrial intelligence, robotics, autonomous systems, information-physical infrastructure and embedded artificial intelligence increasingly require artificial intelligence systems to operate in larger-scale, dynamic and open environments, and put forward higher requirements for adaptability, efficiency, reliability and interpretability.

However, the current artificial intelligence system is still basically based on an independent end-to-end model, and intelligence is mainly obtained by offline training of a large number of static data sets, and then inferred from the deployment time. As the model scale continues to grow, there are more than trillions of parameters (for example, OpenAI GPT-3 with 175B [7], Google PaLM with 540B [8], 405B Meta Llama 3.1 of [9]), the independent end-to-end AI system has encountered unprecedented bottlenecks in data availability, computing scalability, online adaptability and system security. Recent scale law research [10] shows that the growth of model capabilities is increasingly dependent on exponential growth calculations and high-quality training data, and the availability of such data is becoming a major limiting factor. In addition, static pre-training models often find it difficult to continuously adapt to changing environments, long-tail scenarios and interactive real tasks after deployment.

Technically speaking, major artificial intelligence researchers in academia and industry pay more and more attention to these problems. In [11], Richard Sutton presented that scalable intelligence ultimately should come from systems capable of continual learning and adaptation rather than handcrafted priors alone. Further, LeCun et al. [12], [13] also repeatedly pointed out that current auto-regressive language models are insufficient for achieving human-level intelligence due to the fundamental limitations and constraints of language, as they lack persistent world models, planning, memory, and interaction with the physical world. In other words, understanding and interacting with the physical world are essential to achieving human-level intelligence. Similarly, Feifei Li et al. [14] also emphasized that the next generation of AI must move beyond static language intelligence toward embodied and spatial intelligence grounded in interaction with the real world. Furthermore, Geoffrey Hinton and Yoshua Bengio, respectively, have recently raised concerns about the risks and limitations of purely large-scale black-box optimization, which is the foundation of the current AI paradigm, particularly in terms of controllability and interpretability. Bengio et al. in [15] further introduced the Scientist AI and trustworthy AI systems capable of causal reasoning, uncertainty awareness, and safety guarantees, rather than relying solely on predictive scaling. All these emerging viewpoints collectively suggest that AI is approaching a paradigm transition, i.e., from isolated model-centric intelligence toward continuously evolving system-level intelligence. The key challenge is that we must no longer focus solely on building larger models but shift our focus to constructing scalable AI systems capable of continuously generating high-quality adaptive data, collaboratively organizing intelligence, and safely evolving in open environments.

Accordingly, in this paper, to the best of our knowledge, we firstly introduce and comprehensively elaborate a new AI paradigm named Networked Artificial Intelligence (Networked AI), which represents a promising next-generation AI paradigm aimed at addressing the challenges mentioned above. Networked AI refers to a networked intelligent system where heterogeneous AI agents can continuously generate highquality data through mutual interactions, perform autonomous and evolutive optimization of their behaviors in real time, adapt to new environmental conditions, and maintain systemlevel stability, safety, and trustworthiness. As the applicationtier realization of Networking System of AI (NSAI) [16], Networked AI can significantly reduce AI systems’ dependence on massive centralized datasets and exponentially growing compute resources, while continually improving applicationspecific intelligence in continuous, open real-world scenarios.

Compared to traditional standalone end-to-end AI, Networked AI introduces two critical mechanisms absent in current large-model paradigms, i.e., interaction and regulation. Networked AI dynamically extracts high-quality data and generates reliable pseudo-labels or optimization signals during operation [17], [18] by exploiting adaptive interactions guided by prior knowledge, policies, and environmental feedback. Meanwhile, local agent evolution and global system stability can be jointly optimized by leveraging hierarchical regulation mechanisms [19], [20]. These mechanisms can fundamentally improve the following unique features.

• Data efficiency: In Networked AI, traditional static largescale offline datasets will be replaced by continuously generated high-quality interactive data.

• Intelligence efficiency: Networked AI can carry out online evolution and continuous optimization in a dynamic environment, not just relying on static pre-training.

• Interpretability: Networked AI can further transform current AI from black-box optimization to partially structured and explainable gray-box systems through explicit organizational architectures and controllable interactions.

• Scalability: By introducing hierarchical agent organizations capable of decomposing and collaboratively solving arbitrarily complex tasks, Networked AI can overcome the context-window and sequential reasoning limitations of existing end-to-end models.

In Networked AI, the source of intelligence shifts from conventional individual model generalization, to collective and autonomous evolution towards application-specific scenarios. It can present an architectural and system-level embodiment of Online Evolutive Learning (OEL) [18], [21], [22], which deals with the online extraction of high-quality data and the evolutive adaptation of heterogeneous models at the algorithmic level. Technically, traditional standalone AI often exhibits exponential growth in data and computational resource demands as system and task complexities increase. In contrast, Networked AI enables collaborative reuse of knowledge, distributed online optimization, and localized autonomous adaptation, which in turn shifts scaling behavior towards linear growth regimes. It thereby can transform the scaling relationship between system capability and resource demands. In this paper, we discuss the emerging architecture, theory and applications of Networked AI, as well as its industrial impacts and open challenges. Our main contributions are summarized as follows:

• We establish a hierarchical layered architecture for Networked AI (Fig. 1), enabling collaborative optimization among large-scale AI agents. Specifically, in the bottom agent layer, we provide scalable heterogeneous agent communities based on data-driven models; in the middle interaction layer, we discuss online extraction of highquality data through adaptive signal processing guided by prior knowledge and policies; in the top regulation layer, we introduce local-global regulation and control mechanisms maintaining system stability during continual evolution.

• We analyze the scaling relationships between intelligence capability, data and computation demand, and the number of heterogeneous task components, in Networked AI versus standalone end-to-end AI, highlighting the near-linear versus exponential scaling trends; and we also discuss resilient cross-layer security provided by the interpretable layered architecture.

• We further review representative applications in embodied, spatial, and industrial AI, as well as IoT and cellular infrastructures, emphasizing potential transformative gains in the application frontiers.

• Finally, we identify open research challenges in Networked AI, which represent a new scientific paradigm bridging supervised learning, reinforcement learning, continual learning, and distributed intelligence into the next-generation AI systems.

Rather than viewing large foundation models as the endpoint of AI evolution, we argue that those existing large models may instead become foundational components within larger networked intelligent ecosystems. The future of AI may therefore depend less on building even-larger isolated models, but more on constructing continuously evolving, trustworthy, and collaboratively intelligent networked systems. Accordingly, these newly introduced capabilities will establish a novel closedloop AI paradigm integrating system architecture, learning, regulation, security, and verification.

![](images/cb677da4750d76cc8e42a312f66eed4cfb26c29381665b29374c315d85b7f4f6.jpg)  
Fig. 1. Layered architecture of Networked AI.

## II. SYSTEM FRAMEWORK

## A. Definitions

A Networked AI system can be formally abstracted as a time-evolving, spatially distributed dynamical system that fundamentally comprises a universal set of heterogeneous intelligent entities, denoted as $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { N } \}$ . Each agent $a _ { i }$ is an intelligent unit providing a relatively independent function within the overall system. To accommodate disparate architectures and learning paradigms within a common framework, an agent is formally represented as a tuple:

$$
a _ { i } = ( \mathcal { X } _ { i } , \mathcal { Y } _ { i } , \Theta _ { i } , \Pi _ { i } ) ,\tag{1}
$$

where $\mathcal { X } _ { i }$ and $\mathcal { \mathrm { V } } _ { i }$ denote the input and output spaces, respectively. $\Theta _ { i }$ represents the model parameters or rule set. $\Pi _ { i }$ denotes the specific update mechanism. This abstraction establishes a unified systems view for incorporating diverse functional models across the network. These agents do not operate in isolation; rather, the entire networked system is embedded within a non-stationary external environment $\mathcal { E } ( t )$ The spatial interaction topology among these agents at any given time t is represented by a dynamically shifting graph $G ( t )$

Unlike standalone models that map static datasets to fixed parameter spaces, the Networked AI system maintains a distributed global dynamic state S(t), which encapsulates the real-time cognitive and operational conditions of all agents across the network. Driven by networked interactions and bounded by regulation, the macro-level dynamic evolution of the entire system can be abstracted by a generalized global transition operator H, which is given by,

$$
\begin{array} { r } { S ( t + 1 ) = \mathbf { \mathcal { H } } \Big ( S ( t ) , \mathcal { Z } ( t ) , \mathcal { E } ( t ) , { \mathbf { \mathcal { u } } ( t ) } \Big ) , } \end{array}\tag{2}
$$

where $\mathcal { Z } ( t )$ represents the global integrated interaction flow across the dynamic graph $G ( t )$ and u(t) encapsulates the topdown global regulatory control inputs. However, resolving the global operator H directly is computationally intractable due to the massive scale and extreme heterogeneity of the agents. To make this system practically computable and scalable, the macro-level transition is decoupled into localized, autonomous state transitions. Specifically, for an arbitrary agent ${ { a } _ { i } } ,$ its local state transition is governed by its internal dynamic state $s _ { i } ( t )$ which inherently includes its evolvable parameters $\Theta _ { i } ( t )$

$$
\begin{array} { r } { s _ { i } ( t + 1 ) = \mathcal { H } _ { i } \left( s _ { i } ( t ) , \mathcal { Z } _ { i } ( t ) , \mathcal { E } ( t ) , u _ { i } ( t ) \right) , } \end{array}\tag{3}
$$

where $\mathcal { H } _ { i } ( \cdot )$ is the internal update mechanism, $\mathcal { Z } _ { i } ( t )$ is the localized interaction flow filtered from neighboring peers, and $u _ { i } ( t )$ is the localized stability-oriented control input.

Instead of minimizing a static empirical risk, the fundamental objective of Networked AI is to continuously drive the global state $S ( t )$ towards optimal functional consensus and system-wide stability while adapting to the non-stationary environment $\mathcal { E } ( t )$ . To achieve this, the system relies on a hierarchical coordination framework composed of two complementary mechanisms, i.e., 1) a bottom-up interactive learning process: during localized interactions, agents continuously extract high-quality data and formulate evolutive learning objectives by processing information exchanged with their peers. And 2) a top-down regulation process: the control input $u _ { i } ( t )$ provides top-down regulation to constrain local updates and ensure global stability. By seamlessly coupling interactiondriven data extraction with macro-level regulation, the system ensures that localized autonomous evolution stably and robustly serves the collective intelligence without diverging.

To ensure mathematical clarity and consistency throughout the theoretical formulation, a few symbols used in this paper are summarized in Table I. The subscript i denotes an individual agent. (t) denotes the discrete time step.

TABLE I  
NOMENCLATURE OF THE NETWORKED AI FRAMEWORK
<table><tr><td>Symbol Definition / Description</td><td></td></tr><tr><td> $\boldsymbol { A }$ </td><td>Universal set of all heterogeneous intelligent entities</td></tr><tr><td>N</td><td>Total number of agents in the network</td></tr><tr><td>a</td><td>An individual autonomous intelligent agent</td></tr><tr><td> $x , y$ </td><td>Input and output spaces of an agent (respectively)</td></tr><tr><td>Θ</td><td>Internal model parameters or rule set of an agent</td></tr><tr><td>Ⅱ</td><td>Specific update mechanism / learning paradigm of an agent</td></tr><tr><td>t</td><td>Discrete time step index</td></tr><tr><td>ε</td><td>Non-stationary external environment</td></tr><tr><td>G</td><td>Dynamically shifting spatial interaction graph / topology</td></tr><tr><td>S, s</td><td>Global dynamic state of the system, and local dynamic state of an agent</td></tr><tr><td>H, H</td><td>Global transition operator, and local internal update mecha- nism</td></tr><tr><td> $\mathcal { Z } , \mathcal { Z }$ </td><td>Global integrated interaction flow, and localized interaction flow</td></tr><tr><td> ${ \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf } ^ { \mathbf { } } { \mathbf { } } ^ { \mathbf { } } { \mathbf } ^ { \mathbf { } }$ </td><td>Global regulatory control inputs, and local stability-oriented control input</td></tr></table>

## B. Layered Architecture

In order to systematically describe the functional organization of Networked AI, we adopt a three-layer architecture that separates capability building, local evolution and system-level regulation:

1) Agent Layer: It provides the basic capability unit of the system, which is composed of heterogeneous agents with different model structures, learning paradigms and functional specialization. It focuses on local adaptability under representation learning, task-specific modeling and resource constraints.

2) Interaction Layer: It controls the decentralized coordination and local evolution process between agents. It defines the interactive topology $G ( t )$ and realizes evolutive adaptation through neighborhood communication, consistency enforcement and other mechanisms.

The interaction layer is responsible for converting local updates into collective evolution for specific application systems.

3) Regulation Layer: It regulates the behavior of the entire network system by introducing global objectives, a priori constraints and feedback signals, thus providing systemlevel control. Key functions include stability control, convergence adjustment and reducing error propagation in large-scale distributed applications.

The three layers jointly define a hierarchical structure in which local intelligence, interaction dynamics, and global regulation are coherently integrated. This abstraction distinguishes Networked AI from traditional paradigms such as federated learning, multi-agent reinforcement learning, and swarm intelligence, by explicitly incorporating communication-based continual evolution and system-level control into an autonomous intelligence system.

## III. THEORETICAL FOUNDATIONS

## A. Agent Layer: Formation of Heterogeneous Intelligent Agent Community

The Agent Layer provides the fundamental capability units of Networked AI to learn from online data. The system integrates a diverse set of intelligent models with different functional roles and learning capacities instead of assuming homogeneous agents. Agents include foundation models, such as large language models (LLMs) [9], [23]–[25], vision-language models (VLMs) [26]–[29], and vision-language-action (VLA) models [30]–[33]; skill-oriented models which are used for planning, tool utilization, and workflow execution [34]–[36]; traditional perception modules such as Convolutional Neural Networks [1], [2], [37]–[40] and Vision Transformers [41], [42]); conventional decision-making modules like Markov Decision Processes [43] and reinforcement learning agents [44]); and lightweight edge AI models operating under latency, memory, and energy constraints [45], [46]. This heterogeneity allows the system to combine general reasoning abilities with domain-specific skills, forming a rich capability space.

A primary challenge lies in their uniform management and representation since these models exhibit diverse functions, architectures, and update mechanisms. As formally abstracted in Section II-A, Networked AI addresses this by formalizing each functional model as an autonomous agent tuple $a _ { i } = ( \mathcal { X } _ { i } , \mathcal { Y } _ { i } , \Theta _ { i } , \Pi _ { i } )$ within the universal set A (see Eq. (1)). Building upon this formalization, the Agent Layer mainly focuses on organizing these discrete entities into a structured and scalable topology.

When the number of agents becomes large, a flat organization quickly becomes intractable. Placing all agents in a single conceptual plane leads to profound coordination complexity and diminished scalability. To address this, Networked AI employs a scalable, hierarchical agent architecture. Let the Agent Layer consist of L levels,

$$
\mathcal { A } = \bigcup _ { l = 1 } ^ { L } \mathcal { A } ^ { ( l ) } , \qquad \mathcal { A } ^ { ( l ) } = \left\{ \boldsymbol { a } _ { 1 } ^ { ( l ) } , \boldsymbol { a } _ { 2 } ^ { ( l ) } , \dots , \boldsymbol { a } _ { K _ { l } } ^ { ( l ) } \right\} ,\tag{4}
$$

where $a _ { i } ^ { ( l ) }$ denotes the i-th agent at level l, and $K _ { l }$ is the number of agents at that level. For $l \ \geq \ 2$ , each higherlevel agent is defined as a collection of lower-level agents, representing a functional abstraction,

$$
a _ { i } ^ { ( l ) } = \left\{ a _ { j } ^ { ( l - 1 ) } \ : \left| \ : j \in \Omega _ { i } ^ { ( l ) } \right. \right\} ,\tag{5}
$$

where $\Omega _ { i } ^ { ( l ) }$ is the index set of child agents contained within $a _ { i } ^ { ( l ) }$

Depending on their respective functional roles, individual agents within this hierarchy may be optimized using diverse paradigms $( \mathrm { e . g . }$ , supervised learning, reinforcement learning, or explicit skill construction). Regardless of the specific learning paradigm employed, their evolution dynamics follow the unified state update formulation introduced in Eq. (3). Specifically, situated within this hierarchical architecture, the dynamic state evolution of an agent $a _ { i } ^ { ( l ) }$ at level l is explicitly contextualized as,

$$
s _ { i } ^ { ( l ) } ( t + 1 ) = \\mathcal { H } _ { i } ^ { ( l ) } \left( s _ { i } ^ { ( l ) } ( t ) , \mathcal { Z } _ { i } ^ { ( l ) } ( t ) , \mathcal { E } ( t ) , u _ { i } ^ { ( l ) } ( t ) \right) ,\tag{6}
$$

where the general operator $\mathcal { H } _ { i } ^ { ( l ) } ( \cdot )$ governs the internal update, the crucial structural feature introduced by the Agent Layer is the integration of cross-level coupling. In this hierarchy, the localized interaction flow $\mathcal { Z } _ { i } ^ { ( l ) } ( t )$ dynamically fuses both horizontal peer interactions and vertical functional abstractions through a macro-operator $\mathcal F ( \cdot )$

$$
\mathcal { Z } _ { i } ^ { ( l ) } ( t ) = \mathcal { F } \left( \sum _ { k } \alpha _ { i k } ^ { ( l ) } s _ { k } ^ { ( l ) } ( t ) , \mathrm { ~ } \bar { \mathcal { Z } } ^ { ( l ) } \left[ \sum _ { j } \beta _ { i j } ^ { ( l - 1 \to l ) } s _ { j } ^ { ( l - 1 ) } ( t ) \right] \right)\tag{7}
$$

where $\begin{array} { r l } { \sum _ { k } \alpha _ { i k } ^ { ( l ) } s _ { k } ^ { ( l ) } ( t ) } & { { } } \end{array}$ captures the horizontal peer interaction with neighboring agents at the same level l and $\begin{array} { r } { \bar { \mathcal { T } } ^ { ( l ) } [ \sum _ { j } \beta _ { i j } ^ { ( l - 1  \overline { { l } } ) } s _ { j } ^ { ( l - 1 ) } \overline { { ( t ) } } ] } \end{array}$ denotes the vertical functional abstraction from lower-layer child agents. $\alpha _ { i k } ^ { ( l ) }$ represents the interaction strength between peer agents. $\beta _ { i j } ^ { ( l - 1 \dot {  } l ) }$ quantifies the contribution of lower-layer agents to the upper-layer abstraction. Technically, this non-linear mapping function $\mathcal { T } ^ { ( l ) } ( \cdot )$ is strictly deployed to align, compress, and project disparate, low-level multi-modal representations (e.g., pixellevel detections) into the homogeneous algebraic space of the current level l. This formalized abstraction establishes a blueprint for the operational mechanics of the system. In the subsequent sections, the Interaction Layer (Section III-B) explicitly instantiates the macro-operator $\mathcal F ( \cdot )$ by detailing its constituent steps of signal processing and cost formulation, mapping the abstract interaction flow $\mathcal { Z } _ { i } ^ { ( l ) } ( t )$ to its operational data-loss pair $( \mathcal { D } _ { i } ^ { * } ( t ) , J _ { i } ( t ) )$ . Meanwhile, the Regulation Layer (Section III-C) operationally governs the state transition function $\mathcal { H } _ { i } ^ { ( l ) } ( \cdot )$ by injecting the stability-oriented control input $u _ { i } ^ { ( l ) } ( t )$ derived from Lyapunov theory, ensuring that the localized parameter evolution converges robustly toward the global system objective.

B. Interaction Layer: Autonomous Evolution and Coordination by Mutual Interaction

As mentioned above, the Agent Layer provides heterogeneous capabilities, while the Interaction Layer governs how agents undergo local autonomous evolution through adaptive signal processing. This layer essentially defines the knowledge and message exchange mechanisms between agents, ensuring that all interactions are explainable and grounded in prior knowledge, rather than functioning in a black-box manner. Existing signal processing theories/methods provide the theoretical foundation for this interpretable exchange. For instance, optimal estimation techniques such as Kalman filter [47] and Wiener filtering [48] are employed to denoise inter-agent signals, while adaptive algorithms like Least Mean Squares (LMS) and Recursive Least Squares (RLS) [49] allow agents to adjust to non-stationary environments. Furthermore, Graph Signal Processing (GSP) [50] is utilized to analyze information flow across the interaction graph $G ( t )$ , and Bayesian inference [51] enables the fusion of signals with applicationspecific priors. These methods act as functional abstractions that ensure every piece of exchanged information can be analyzed and verified based on the system’s prior knowledge, transforming raw interactions into a structured dialogue. Recent studies have further shown that structural knowledge of sensing can be explicitly embedded into online optimization, improving both sample efficiency and explainability under dynamically changing conditions [52]. This observation supports the Interaction-Layer design principle that prior-guided adaptive signal processing can provide an interpretable foundation for evolution.

An essential function of the Interaction Layer is to continuously extract high-quality data and generate a corresponding cost function for every interactive agent. Mathematically, to instantiate the interaction flow defined in Eq. (7), an agent continuously utilizes the aforementioned signal processing tools to filter the raw information exchanged with its active neighbors over the dynamic graph $G ( t )$ . This filtering process yields a reliable, denoised network-derived actual signal, denoted as $y _ { i } ( t , G ( t ) ) \in \mathcal { V } _ { i }$ , effectively grounding the interaction within the agent’s predefined output space. The explicit notation of both t and $G ( t )$ formalizes a critical spatiotemporal coupling, i.e., it emphasizes that this consensus signal depends simultaneously on the continuous temporal evolution over time t of the incoming data streams and the structural constraints imposed by the dynamically shifting network topology $G ( t )$ . Concurrently, the agent interacts with the non-stationary environment $\mathcal { E } ( t )$ and monitors the residual error between its own internal prediction $\hat { y } _ { i } ( t ) \ \in \ \mathcal { V } _ { i }$ and this interactive observation. A statistical shift or unfamiliar scenario induced by the evolving environment $\mathcal { E } ( t )$ is detected when this discrepancy exceeds a predefined threshold τ, i.e., $| | \hat { y } _ { i } ( t ) - y _ { i } ( t , G ( t ) ) | | > \tau$ . Triggered by this environmentinduced discrepancy, the agent initiates a targeted adaptation process. Using a localized extraction function Ψ, it processes this error to identify a set of high-quality data $\mathcal { D } _ { i } ^ { * }$ and formulate a specific local cost function $J _ { i } .$ . We formalize this interaction-driven extraction process as follows,

$$
( \mathcal { D } _ { i } ^ { * } ( t ) , J _ { i } ( t ) ) = \Psi ( \hat { y } _ { i } ( t ) , y _ { i } ( t , G ( t ) ) , \mathcal { E } ( t ) , \tau ) .\tag{8}
$$

This abstraction ensures that the system selectively filters interaction signals over $G ( t )$ to isolate high-fidelity components that reflect the current state of $\mathcal { E } ( t )$ . For example, a detailed illustration of interaction-driven joint optimization process is given in [18].

This interaction-driven update mechanism enables the system in transition from offline generalization to online evolutive adaptation. By executing targeted data extraction and parameter updates solely upon detecting statistical novelties in the signal space, as well as the prior knowledge embedded in the adaptive signal processing for agent interactions, the Interaction Layer supports localized feature processing and avoids unrestricted system-wide Cartesian expansion, providing the structural basis for the near-linear scaling regime.

## C. Regulation Layer: System-Level Governance by Online Distributed Regulation

The Regulation Layer establishes global mechanisms to coordinate collective behavior and ensure long-term stability within the Networked AI architecture. The theoretical foundations for this layer are rooted in theory of system control, particularly Lyapunov stability analysis [53], Model Predictive Control [54], and robust control strategies. Lyapunov method provides a mathematical framework for proving the convergence of dynamic systems by constructing a scalar energy function, while robust control solves the problem of performance margin under environmental uncertainty. In Networked AI, these theories allow a formal assessment of the evolutive path of heterogeneous agent communities and prevent systemwide differences.

Traditional data-driven models are often known as blackbox systems. Due to the absence of interpretable inputoutput mappings, it is very difficult to apply system control methodologies directly to the training process. However, the proposed Networked AI addresses this limitation by utilizing the Interaction Layer to convert raw signals into interpretable relationships grounded in prior knowledge. This structural transformation effectively enables the intelligent system to be treated as a gray-box that can be formally described and regulated. The Regulation Layer can then mediate the relationship between local autonomous evolution and global system requirements, by incorporating physical feedback and high-level constraints into the governance cycle.

We hereby introduce a scalar Lyapunov candidate function $V ( S ( t ) )$ representing the global energy or collective error of the overall system. To align with the hierarchical architecture defined in the Agent Layer, the global state $S ( t )$ is formulated as the universal set of all dynamic agent states across the network, i.e., $S ( t ) = \Bigl \{ s _ { i } ^ { ( l ) } ( t ) \vert 1 \le l \le L , 1 \le i \le K _ { l } \Bigl \}$ , where $s _ { i } ^ { ( l ) } ( t )$ is introduced in Eq. (6). Specifically, $V ( S ( t ) )$ acts as the dynamic evaluation metric for the entire system. As the collective minimization of local costs $J _ { i }$ drives the descent of the global summation term, the system-level constraints are encoded within the energy landscape of $V ( S ( t ) )$ . However, continuously computing the exact global variation requires centralized aggregation, which can be computationally prohibitive and may violate decentralized constraints of some Networked-AI applications. To resolve this, the Regulation Layer may assume that the global energy function is additively separable or can be locally approximated, i.e.,

$$
V ( S ( t ) ) \approx \sum _ { i } V _ { i } ( s _ { i } ( t ) , \mathcal { Z } _ { i } ( t ) ) , \quad 1 \leq i \leq K .\tag{9}
$$

Consequently, each agent evaluates a predicted local-energy variation under the unregulated candidate update, denoted as $\Delta V _ { i } ( t )$ which estimates its individual contribution to the global system dissipation based solely on its state and neighborhood interaction flow. To approach asymptotic stability, the stability-oriented control input $u _ { i } ^ { ( l ) } ( \dot { t } )$ in Eq. (6) is operationally instantiated as a local regulation operator, i.e., $\mathbf { \bar { \Psi } } u _ { i } ^ { ( l ) } ( t ) : = \Gamma _ { i } ( t )$ . This operator is formulated to set up local passivity, which can collectively achieve system-wide dissipation:

$$
\Gamma _ { i } ( t ) = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f } \ \Delta V _ { i } ( t ) \leq 0 } } \\ { { \exp ( - \kappa \Delta V _ { i } ( t ) ) , } } & { { \mathrm { i f } \ \Delta V _ { i } ( t ) > 0 } } \end{array} \right.\tag{10}
$$

where κ is a predefined positive scaling coefficient. For an agent parameterized by $\Theta _ { i }$ (omitting the layer superscript (l) for notational simplicity), the regulated parameter update is subsequently expressed as follows:

$$
\Theta _ { i } ( t + 1 ) = \Theta _ { i } ( t ) - \Gamma _ { i } ( t ) \cdot \eta \nabla _ { \Theta _ { i } } J _ { i } ( \mathcal { D } _ { i } ^ { * } , \Theta _ { i } ) ,\tag{11}
$$

where $\Gamma _ { i } ( t ) \ \in \ ( 0 , 1 ]$ adjusts the effective learning rate η. This regulated update law provides a concrete and parametric realization of the abstract state transition function $\mathcal { \bar { H } } _ { i } ^ { ( l ) } ( \cdot )$ by treating the model parameters as the core evolvable component of the agent’s dynamic state, i.e., $\Theta _ { i } ( t ) \subseteq s _ { i } ^ { ( l ) } ( t )$ . Specifically, the internal state transition is driven by the local cost $J _ { i }$ and the extracted data $\mathcal { D } _ { i } ^ { * }$ , which directly operationalize the abstract interaction flow $\bar { \mathcal { Z } } _ { i } ^ { ( l ) } ( t )$ . This implicitly encodes the influence of horizontal peer interaction and vertical function abstraction from the local neighborhood of the agent. It in principle ensures that the localized, heterogeneous parameter evolution $\Theta _ { i }$ may converge toward the global system objective by incorporating the environmental feedback through $\mathcal { D } _ { i } ^ { * }$ and coupling it with the Lyapunov regulation factor $\Gamma _ { i } ( t )$ , thereby maintaining the robustness of the collective intelligence in non-stationary settings (Theorem 3, Appendix B).

## D. Efficiency of Collective Intelligence

While the preceding sections have established the theoretical foundation of the proposed three-layer Networked AI architecture, we hereby further connect the layered architectural design to its complexity scaling analysis from both theoretical and experimental perspectives.

From a theoretical perspective, a standalone end-to-end system needs to jointly process heterogeneous tasks, as comparable to those from all N agents in Networked AI. Therefore, N can virtually indicate the scale of overall task complexity or system capability for both systems. This global coupling leads to an exponential training-data requirement of $\bar { \Omega ( c ^ { N } ) }$ for standalone systems (Theorem 1, Appendix A), where c denotes the independent feature complexity from an individual task.

In contrast, the hierarchical multi-agent community, as specified in the Networked-AI architecture and theory, is able to structurally reduce the system coordination depth, effectively lowering the overall task degree of freedom from $O ( N )$ as in standalone systems, to ${ \cal O } ( \log N )$ in Networked AI, where log N is the number of hierarchical levels. Therefore, the corresponding training data requirement can be reduced to $O ( N \cdot c ^ { \log N } )$ ), which can actually be close to a near-linear scaling of N in Networked AI (Theorem 2, Appendix A). The reduction is dependent on the system design with embedded depth and prior knowledge in the Networked-AI architecture.

The above theoretical views give a scaling comparison: when the overall task complexity or system capacity increases, the resources required by standalone end-to-end system incurs superlinear or exponential growth, and Networked AI can approach near-linear growth through localized agents decomposition learning and hierarchical coordination. In order to verify the theoretic comparison, we have built a controlled experimental environment in which the model scale, application scale and overall task complexity can change independently. Under the matching task conditions, we compared the standalone AI configuration and the Networked AI configuration, and then fit the resource-scaling curve.

Because the Interaction Layer extracts task-relevant samples and only activates localized updates for deficient components, redundant data acquisition and global retraining are reduced. This on-demand and evolutive adaptation mechanism improves data and intelligence efficiency in a changing environment. Therefore, from the experimental perspective, we evaluated the computational cost of convergence under different models and application settings as an empirical proxy for interaction or data demand.

In the previous study, sensing-control decoupling and OEL experiments were introduced [55], and we extended the original local feedback loop to a hierarchical systems containing multiple heterogeneous agents. We employ a configurable grid-world in which agents navigate toward a target while avoiding known obstacles and dynamically introduced unknown obstacles. The following experiments include 6 heterogeneous agents, 5 model scales, 8 application scenarios, and the spatial size and obstacle density gradually increase. Eliminate unknown obstacles from the initial training distribution, simulate the non-stationary environmental displacement $\mathcal { E } ( t )$ and the novelty outside the distribution. In each time step, the agents receive a local observation, select a discrete movement action, and get a sparse rewards for penalizing collision and rewarding successful navigation. This experiment is mapped to the proposed three-layer architecture, as follows:

• Agent Layer: Assign heterogeneous perception and control functions to specialized agents. In the experimental implementation, these roles are instantiated by YOLOv5- based perception agents and DQN-based control agents. However, the architecture is not limited to these specific models and can accommodate other perception and decision-making modules.

• Interaction Layer: Consistent with the theoretical formula, the discrepancy signal is defined as $\Delta _ { i } = | R _ { p } -$ $R _ { a } |$ . Here, $R _ { p }$ is the predicted reward, and $R _ { a }$ is the actual reward, which correspond to ${ \hat { y } } _ { i } ( t )$ and $y _ { i } ( t , G ( t ) )$ in the interaction layer, respectively. When $\Delta _ { i } > \tau .$ , the system extracts the corresponding interaction sample and stores it in the localized dataset $\mathcal { D } _ { i } ^ { * }$

• Regulation Layer: The localized buffer triggers an asynchronous update $\Theta _ { i }$ only for the deficient component, avoiding unnecessary global backpropagation and interference with unaffected modules.

We measured the total FLOPs accumulated from experimental initialization to policy convergence. When an agent reaches the target, it does not collide with known or unknown obstacles within the predefined number of consecutive episodes, that is, convergence is achieved. Since the data sample is generated through the interaction between the agent and the environment, the convergent calculation cost can be used as an empirical proxy for interaction and data requirements.

For data visualization, the observations that share the same model scale, application scale or overall task complexity are grouped into a box plots, which summarize the median, quartiles and dispersion of different experimental configurations. The vertical axes for model and application scales are plotted logarithmically, whereas a linear scale is adopted for the overall task complexity (agent numbers). For model and application scales, the overall scaling trends are obtained by fitting the power-law model $F = k X ^ { \alpha }$ through log-log leastsquares regression, where $X = P$ denotes the model parameter count and $X \ = \ E$ denotes the application scale level. For overall task complexity, represented by the agent number $N ,$ the arithmetic mean FLOPs at each N are fitted using the exponential-linear model $F = a e ^ { b N } + c N + d$ through nonlinear least-squares optimization. The box plots characterize the variability in computational costs across heterogeneous experimental configurations, while the fitted curves capture the global resource-scaling behavior of standalone and Networked AI.

Fig. 2a shows the relationship between model scale and computational cost. The horizontal axis represents the actual number of model parameters P. For each model scale, the box plot summarizes the distribution of convergence costs in different application scenarios, and the heterogeneous agent N is only set to 2 and 3 to isolate the impact of the model scale. The global power-law fits are:

$$
F _ { \mathrm { s t a n d a l o n e } } ( P ) \approx 1 . 9 \times 1 0 ^ { - 3 } P ^ { 1 . 2 1 } ,\tag{12}
$$

and

$$
F _ { \mathrm { n e t w o r k e d } } ( P ) \approx 8 . 1 \times 1 0 ^ { - 3 } P ^ { 1 . 0 4 } .\tag{13}
$$

The exponent of standalone AI is super-linear, whereas Networked AI remains close to linear. This result indicates that increasing model capacity imposes a growing global reoptimization cost on the standalone system, while localized updates allow Networked AI to limit the additional interaction and data demand.

Fig. 2b depicts the effect of application scale, where the number of heterogeneous agents N is still set at 2 and 3 only, to isolate the impact of application. For a grid world of size $H \times W$ , we first calculate an application score

Computational Cost vs. Model Scale  
![](images/aa93d313aa48e8f4736716899d9e48e1b67c5815c91ac8ddff6133de6c2007aa.jpg)  
(a)

Computational Cost vs. Application Scale  
![](images/f33cccdbe31bf45fae2494617cd3df54afb8a0e7014f217913e67d23882a81ca.jpg)  
(b)

Computational Cost vs. Overall Task Complexity  
![](images/8d66a1a0b0a82409a72151a69b30a177858c83c28e634bb8104a10dc61a91387.jpg)  
(c)  
Fig. 2. Computational cost to convergence under different model, application and overall task complexity scales. The box plots summarize the variability across heterogeneous experimental configurations, while the solid and dashed curves represent the global power-law or exponential fits for standalone AI and Networked AI.

$$
\Phi = ( N _ { U } + N _ { O } ) \sqrt { H \times W } ,\tag{14}
$$

where $N _ { U }$ and $N _ { O }$ denote the numbers of unknown objects and known obstacles, respectively. The evaluated scenarios are ordered according to Φ and mapped to the application-scale levels $E \in \{ 1 , \ldots , 8 \}$ . At each level, the box plot summarizes the convergence costs obtained under different model scales and agent configurations. The global power-law fits are

$$
F _ { \mathrm { s t a n d a l o n e } } ( E ) \approx 2 . 4 \times 1 0 ^ { 4 } E ^ { 2 . 5 5 } ,\tag{15}
$$

and

$$
F _ { \mathrm { n e t w o r k e d } } ( E ) \approx 7 . 5 \times 1 0 ^ { 4 } E ^ { 1 . 2 9 } .\tag{16}
$$

The substantially smaller exponent of Networked AI indicates that its training cost is less sensitive to increasing application scale.Since $N \in \{ 2 , 3 \}$ , the theoretical model predicts that globally coupled standalone learning may exhibit an effective dependence between $O ( c ^ { 2 } )$ and $O ( c ^ { \bar { 3 } } )$ , whereas Networked AI should exhibit a lower-order growth trend through functional decomposition and localized optimization. The empirical fitting, $\bar { E } ^ { 2 . 5 5 }$ for standalone AI and $E ^ { 1 . 2 9 }$ for Networked AI, are matching with the theoretical base: the standalone result lies within the expected near-quadratic to near-cubic range, while the Networked AI shows substantially milder growth.

Fig. 2c represents computational cost as a function of the overall task complexity, indexed by the heterogeneous agent number $N \in \{ 2 , 3 , 4 , 5 , 6 \}$ . For the standalone systems, each N value represents a comparable collection of heterogeneous tasks that must be processed jointly. For each $N ,$ , the box plots summarize the convergence cost of five model scales and eight application scales. The fits are

$$
F _ { \mathrm { s t a n d a l o n e } } ( N ) \approx 4 . 8 2 2 4 \times 1 0 ^ { 6 } e ^ { 0 . 5 2 0 7 7 N } ,\tag{17}
$$

and

$$
F _ { \mathrm { n e t w o r k e d } } ( N ) \approx 2 . 3 6 4 5 \times 1 0 ^ { 5 } N + 2 . 0 7 6 3 \times 1 0 ^ { 6 } .\tag{18}
$$

The fitted results reveal distinct computational scaling trends. Standalone AI exhibits exponential growth with an exponent of 0.52077, whereas Networked AI follows an approximately linear trend, with an incremental cost of only $2 . 3 6 4 5 \ \times \ 1 0 ^ { 5 }$ FLOPs per additional agent. As N increases from 2 to 6, the fitted computational cost increases by a factor of $\begin{array} { r l } { F _ { \mathrm { s t a n d a l o n e } } ( 6 ) / F _ { \mathrm { s t a n d a l o n e } } ( 2 ) } & { { } = } \end{array}$ $e ^ { 0 . 5 2 0 7 7 ( 6 - 2 ) } \approx 8 . 0 3$ for standalone AI, compared with only $F _ { \mathrm { n e t w o r k e d } } ( 6 ) / F _ { \mathrm { n e t w o r k e d } } ( 2 ) ~ = ~ 3 . 4 9 5 0 / 2 . 5 4 9 2 ~ \approx ~ 1 . 3 7 ~$ for Networked AI.

Under the evaluated configurations, hierarchical decomposition and localized optimization in Networked AI substantially reduce the observed growth rate of data and computational demand. Together with Appendix A, the results support a conversion of exponential-to-near-linear scaling, which can reduce the order of magnitude for data and computational resource demand in complicated tasks.

## E. Cross-layer and Resilient Security

Networked AI transforms deep-learning based standalone AI from a traditional large black-box system into a gray-box where all interactions and regulations over the networked AI agents can be clearly interpreted. Following the principle that ”security emerges from architecture”, a multi-level defense method is required, including single agent reinforcement, collective anomaly recovery ability and system-level recovery. This involves the entire life cycle of protecting data, models and behaviors under dynamic and potentially unfavorable conditions. By integrating security into all layers of the architecture, Networked AI can achieve enhanced resilience, viability and robustness against known and unpredictable interruptions.

The transition from an independent end-to-end model to Networked AI has also introduced a different threat scenario. When an independent model presents a larger attack surface, Networked AI is composed of heterogeneous, interactive and evolving Agents, presenting a dynamic and multi-dimensional vulnerability space. For example, conventional threat modeling frameworks were designed for systems with static computational graphs, fixed data flows, and centralized control logic, which lack the expressiveness to model threats arising from agent autonomy, emergent inter-agent behavior, and the online adaptation mechanisms. Recent work such as MAESTRO [56] represents an advance in agent-centric threat modeling, but is still scoped primarily to individual agents instead of the systemic, cross-layer threat dynamics inherent to Networked AI.

At the Agent Layer, each agent is represented by the tuple $a _ { i } \ = \ ( \mathcal { X } _ { i } , \mathcal { Y } _ { i } , \Theta _ { i } , \Pi _ { i } )$ in Eq. (1) and is continuously optimized using online data. Its input and output spaces, model parameters, learning mechanism, and local state transition therefore can constitute attack surfaces.

• Adversarial manipulation: an adversary may perturb inputs in $\mathcal { X } _ { i }$ or the observations used to produce ${ \hat { y } } _ { i } ( t )$ , causing erroneous predictions or decisions. This disturbance may further pollute the difference test and extracted data and cost pairs $( \mathcal { D } _ { i } ^ { * } ( t ) , J _ { i } ( t ) )$ in Eq. (8), thus turning the inferred time attack into harmful online updates.

• Model tampering/poisoning: malicious modification of the model parameters $\Theta _ { i }$ , the learning mechanism $\Pi _ { i } ,$ or the extracted dataset $\mathcal { D } _ { i } ^ { * } ( t )$ can corrupt the local transition $\mathcal { H } _ { i } ( \cdot )$ in Eqs. (3) and (6). Since $\Theta _ { i }$ is updated through the equation (11), the gradient of poisoning or the local target of manipulation may gradually reduce the professional ability of the agent.

• Evasion attacks: the opponent may carefully design the input that leads to the wrong output, while keeping $\| \hat { y } _ { i } ( t ) - y _ { i } ( t , G ( t ) ) \|$ is lower than the threshold value $\tau$ in the formula (8). Therefore, the attack can circumvent novelty detection and prevent the target adaptive mechanism from being activated.

At the Interaction Layer, the principal attack surface is the localized interaction flow $\mathcal { Z } _ { i } ^ { ( l ) } ( t )$ in $\mathrm { E q . } \ ( 7 )$ , which aggregates horizontal peer states and vertical functional abstractions over the dynamic topology G(t).

• Malicious knowledge injection: compromised agents may transmit falsified states $s _ { k } ^ { ( l ) } ( t )$ or lower-level representations $s _ { j } ^ { ( l - 1 ) } ( t )$ . Through the weighted aggregation coefficients $\alpha _ { i k } ^ { ( l ) }$ and $\beta _ { i j } ^ { ( l - 1  l ) }$ in Eq. (7), these signals can contaminate $\mathcal { Z } _ { i } ^ { ( l ) } ( t ) ^ { \dagger }$ and subsequently the extracted data and objective in Eq. (8).

• Consensus subversion: coordinated malicious agents may manipulate the interaction graph G(t), dominate an agent’s neighborhood, or bias the interaction weights $\alpha _ { i k } ^ { ( l ) }$ This can distort the peer-aggregation term in Eq. (7), allowing adversarial information to appear as a locally consistent consensus signal $y _ { i } ( t , G ( t ) )$

• Trust exploitation: an adversary may compromise neighbor admission or validation mechanisms and enter the neighborhood $_ { \mathrm { o r } }$ child-agent set $\Omega _ { i } ^ { ( l ) }$ in Eq. (5). Once admitted, an untrusted agent can influence both the hierarchical abstraction in Eq. (5) and the horizontal or vertical information flows in Eq. (7).

At the Regulation Layer, attacks target the global governance variables $u ( t )$ and $u _ { i } ^ { ( l ) } ( t )$ in Eqs. (2), (3), and (6), as well as the stability-oriented regulation mechanism in Eqs. (9)–(11).

• Control-loop disruption: an adversary may falsify the estimated local energy variation $\Delta V _ { i } ( t )$ or manipulate the regulation factor $\Gamma _ { i } ( t )$ in Eq. (10). The resulting corrupted control input can alter the effective update step in Eq. (11), weakening or reversing the intended regulation of local adaptation.

• Risk-propagation exploitation: failures originating from a compromised local state $s _ { i } ( t )$ or interaction flow $\mathcal { Z } _ { i } ( t )$ may propagate through the hierarchical composition in Eqs. (5)–(7). Because the global energy in $\operatorname { E q . } \left( 9 \right)$ is only locally approximated by the collection of $V _ { i } ,$ coordinated local failures may also exploit cross-agent dependencies that are not captured by local monitoring.

• Constraint manipulation: an adversary may corrupt the environmental feedback $\mathcal { E } ( t )$ , global constraints, or physical priors entering the transition models in Eqs. (2), (3), and (6). Such manipulation can also bias the dataextraction function $\Psi ( \cdot )$ in Eq. (8), causing otherwise legitimate regulation and adaptation mechanisms to optimize toward an incorrect system objective.

These gaps stimulate a cross-layer security framework, which is structurally consistent with the three-layer architecture of Networked AI, adapts to online and unstable evolution, and can coordinate threat information across architecture boundaries. At the Agent Layer, threat modeling consistent with the architecture should solve the problems of confrontational input, indirect prompt injection, unauthorized operation, model tampering and memory poisoning [56]–[58]. Separating trusted control flows from untrusted data flows can further prevent external content from changing the agent’s execution logic or protected information [59]. At the Interaction Layer, zero-trust identity governance should authenticate participating agents and enforce task-scoped communications [60]. Byzantine-robust aggregation and protocol-level protection should further limit the influence of compromised agents, falsified messages, and over-privileged communication on collective learning [61]–[63]. At the Regulation Layer, trusted monitoring, anomaly isolation, fallback, and rollback should prevent falsified local signals and agent-level failures from propagating into system-wide instability. Because static defenses may fail against adaptive adversaries, the resulting framework should be evaluated under adaptive attacks rather than fixed attack patterns [64]. Together, these mechanisms of cross-layer and resilient security are being developed to jointly form a unified defense across the life-cycle of Networked AI, contributing to a trusted online evolutive system.

## IV. PARADIGM MIGRATIONS

The development of Networked AI is closely related to many research directions such as large-scale foundation models, distributed intelligence, continual learning, multi-agent collaboration and emergency collective systems. Although these fields are largely independent, they together provide an important technical basis for Networked AI to operate in an open and dynamic environment (Table II).

## A. Foundation Models and Standalone AI Systems

The latest progress of deep learning is mainly driven by independent large-scale training models. AlexNet [1], ResNet [2], Transformer [3] lay the foundation for modern AI systems. Subsequent developments in large-scale language models and multimodal learning, including GPT series models [7], CLIP [5] and PaLM [65], show excellent ability in language understanding, reasoning and generative intelligence.

Despite the success of standalone AI systems, they are increasingly limited in terms of scalability, adaptability and deployment efficiency, and the improvement of performance relies heavily on the exponential increase in computing resources and training data. In addition, offline pre-training paradigms often struggle in non-stationary environments that require continual adaptation, application-specific customization and online interaction.

These limitations promote the transition from standalone models to distributed and evolving intelligent systems.

## B. Distributed AI and Multi-Agent Systems

Distributed artificial intelligence (DAI) and Multi-Agent Systems (MAS) provide an early theoretical basis for distributed intelligent systems [66], [67]. In these examples, multiple autonomous agents collaborate through coordination mechanisms to solve complex tasks that are difficult for a single agent to complete.

The existing research mainly focuses on decentralized optimization, consensus dynamics, task allocation and game theory coordination [68], [69]. In recent years, distributed multi-agent reinforcement learning (MARL) systems [70], [71] has received wide attention in the large-scale autonomous system. These methods provide important insights for distributed decision-making and local-to-wide coordination.

However, most existing MAS and MARL frameworks mainly focus on predefined collaboration goals in a relatively static environment. Problems such as online evolutive adaptation, autonomous ability growth, dynamic structure reconfiguration and large-scale heterogeneous intelligence have not been fully studied.

## C. Federated and Distributed Learning

Federated Learning (FL) introduces a distributed learning paradigm in which multiple Agents collaborate to train models without directly sharing raw data [72]–[75]. FL has significantly improved the ability of privacy protection and distributed deployment, making it highly relevant to edge intelligence and large-scale networked systems.

Recent research has extended FL to distributed and asynchronous settings [76], [77], personalized federated learning [78], [79], and federated reinforcement learning [80], [81]. These studies involve communication efficiency, non-IID data distribution and distributed optimization stability.

However, federated learning mainly focuses on collaborative parameter aggregation, rather than autonomous system evolution. Most FL systems still rely on relatively fixed model structures and predetermined training goals, and lack mechanisms for continual adaptation, structural self-reorganization and sudden collective intelligence.

## D. Continual Learning and Online Adaptation

Continual learning aims to enable AI systems to constantly acquire new knowledge while maintaining their previous learning ability [82], [83]. The research on lifelong learning, online adaptation and continual reinforcement learning investigated catastrophic forgetting, knowledge transfer and longterm adaptation [84]–[86].

Recent studies have begun to explore distributed continual learning and collective lifelong intelligence [87]. In particular, Shared-Experience Lifelong Learning (ShELL) system proposes collaborative knowledge accumulation between distributed agents operating at the edge. These advances show that distributed intelligent systems can achieve continual adaptation through experience sharing.

However, existing continual learning frameworks generally focus on isolated learning processes or small-scale collaborative systems, with the target problem primarily on adaptation for generalization, instead of the evolution for applications as in Networked AI. Key missing parts can include a unified framework integrating distributed coordination, autonomous data generation, online structural evolution, and system-level regulation.

## E. Collective Intelligence and Emergent Systems

Collective intelligence and swarm intelligence study how global intelligent behaviors emerge from local interactions among distributed agents [88], [89]. Research in complex adaptive systems, self-organization, and emergent communication has demonstrated that decentralized interactions can generate sophisticated system-level behaviors [90].

These ideas have inspired important developments in swarm robotics, distributed optimization, and cooperative AI [91]. More recently, emergent collaborative reasoning among learning agents have become active research topics in large-scale AI systems [90].

Nevertheless, the foundations of emergent intelligence in large-scale AI networks remain insufficiently understood. Key missing problems can include the formal characterization of collective intelligence, stability of evolving interaction structures, and the relationship between local adaptation and global system behavior.

## V. APPLICATION FRONTIERS

Now, we will study how the Networked AI architecture is mapped to several application areas where data and intelligent efficiency have become major bottlenecks, and discuss what layered design provides in each situation.

## A. Intelligent Internet of Things

The deployment of the Internet of Things across ecological surveillance, smart infrastructure, and industrial sensing produces heterogeneous, noisy and rarely static data flows. In practice, what complicates these data streams is not so much the perceived uncertainty itself as the operational constraints surrounding it: tight energy budgets, intermittent connections, and local data governance rules that limit the possible dissemination of raw data. It is difficult for independent examples to adapt to such a setting. Streaming the original measurement data to the cloud is expensive in terms of bandwidth and energy, and retraining each site in isolation will blind the system to cross-site drift and events often observed by few single nodes. Networked AI regards each edge node as an autonomous Agent-Layer unit, which perceives, filters and adapts locally, and only shares bounded summaries with its neighbors.

On-site sonar monitoring of wild river ecosystems is a typical example. Under the sparse network coverage, the moving targets of the underwater target must be tracked [92]. Agents who handle these streams can be responsible for fish counting, trajectory tracking or abnormal detection, and they do not require a global view. The governance dimension is equally binding: community-led ecological projects, infrastructure operators and industrial facilities are often unwilling or legally unable to gather raw data, parameters or embedding. By exchanging compact prototypes and sound retrieval signals, agents can compare observations across domains, while the underlying data remains local [93].

The signal processing tool specifically instantiates the Interaction Layer. Kalman filtering, temporal smoothing, background modeling and graph signal processing exceeding $G ( t )$ allow each agent to suppress noise before sharing any content and only retain evidence related to the task. On top of this, the Regulation Layer adjusts the update steps through $\Gamma _ { i } ( t )$ , and reallocates sensing, computation and energy when the network degrades or the environment changes suddenly. This brings specific rather than generic: edge analysis within the scope of energy and connectivity budgets, and the exchange protocol for maintaining local management data.

## B. Embodied Intelligence

Embodied systems such as autonomous robots, drones and collaborative manipulators run in a physically dynamic and partially observable environment, in which the strategy of offline training will be downgraded once it deviates from the training distribution. Centralized control makes the problems in multi-agent settings more complicated, because communication delays and environmental changes leave little room for anything other than device adaptation. The Interaction Layer directly solves this problem: once the difference $\parallel \hat { y } _ { i } ( t ) -$ $y _ { i } ( t , G ( t ) ) \| > \tau$ appears in the physical interaction process, it folds the a priori in the fields of kinematic limit and contact dynamics as the cost function $J _ { i }$ . Data efficiency is decisive here, because under real physical conditions, highquality interactive data is scarce and expensive.

The multimodal case expands this picture. The agent for perception through visual, audio, touch and proprioception requires the Agent Layer to host vision-language-action models and lightweight edge perception, all of which represent $\left( \mathcal { X } _ { i } , \mathcal { Y } _ { i } , \Theta _ { i } , \Pi _ { i } \right)$ . Then, cross-modal alignment becomes the responsibility of the Interaction Layer: adaptive filtering coordinates the asynchronous stream into a consistent representation, so that language instructions can be transformed into action sequences based on the actual content of the agent’s perception. Human-robot interaction is the most difficult to emphasize this cycle, because scenario understanding, voice commands and tactile feedback must be coordinated under a tight delay budget.

The collaboration between agents is the most direct contribution of the network structure. Task-related experience flows along the dynamic diagram $G ( t )$ , so skill acquisition accelerates without any synchronous global updates, and the Regulation Layer processes the formation again when the agent fails or the task is reassigned through the Lyapunovbased actor $\Gamma _ { i } ( t )$ . Configuration and load balancing. The modular structure brings a practical consequence: in securitycritical deployments, failures can be attributed to specific agents and their audited behavior, and a single strategy cannot provide such transparency.

## C. Spatial Intelligence

Spatial distribution perception, including synchronous positioning and mapping, collaborative scene reconstruction and multi-viewpoint object recognition, puts forward consistency and bandwidth requirements that are difficult to meet by standalone methods as the scene grows. Each agent only observes a part of the environment, so a coherent global map must be combined from these local views. Networked AI organizes the participating perception subjects on an adaptive interaction diagram, and verifies and integrates its observation results through the knowledge exchange protocol of the Interaction Layer. Consistency is a concern of the Regulation Layer: it enforces geometric and semantic constraints between agents so that contradictory local mappings do not damage the shared scenario pictures. Due to the different abilities of agents, in the case of a LiDAR-based depth estimation and another visual-language semantic labeling, they can form task-specific subnetworks according to the hierarchy of Eq. (5), and the generated topology and Trace coverage and bandwidth, not pre-fixed.

The second requirement is about the temporal consideration. The real environment will not stand still: the map is outdated, the observation value conflicts, and the agent population itself moves, so the system must be constantly updated, instead of converging and staying unchanged. Therefore, the joint evolution of agent representation and interaction structure becomes a defining attribute of large-scale spatial intelligence, not an accidental feature, which is most important when coverage, uncertainty and resources change during operation. Compared with the overall mapping that relies on global aggregation, the advantage here is to maintain consistency under changes while reducing data and calculations.

TABLE II  
PARADIGM MIGRATIONS TOWARD NETWORKED AI.
<table><tr><td>Paradigm</td><td>Intelligence Source</td><td>Data Source</td><td>Adaptation</td><td>Coordination</td><td>Scalability</td></tr><tr><td>Expert Systems</td><td>Human Rules</td><td>Knowledge Bases</td><td>None</td><td>None</td><td>Low</td></tr><tr><td>Deep Learning</td><td>Model Parameters</td><td>Labeled Datasets</td><td>Offline</td><td>None</td><td>Moderate</td></tr><tr><td>Foundation Models</td><td>Large-scale Pretraining</td><td>Web-scale Data</td><td>Limited</td><td>None</td><td>Compute-Bounded</td></tr><tr><td>Multi-Agent AI</td><td>Agent Cooperation</td><td>Environment Interaction</td><td>Partial</td><td>Explicit</td><td>Moderate</td></tr><tr><td>Networked AI</td><td>Collective Evolution</td><td>Knowledge and Operation Data</td><td>Online Evolutive</td><td>Adaptive + Regulated</td><td>Hierarchical and Near-linear</td></tr></table>

## D. Industrial Intelligence

Industrial AI includes a tightly coupled process across edge devices, factory controllers, and cloud infrastructure, in which delay-sensitive control decisions must coexist with long-horizon optimization goals. Conventional standalone AI schedulers lack the flexibility to respond to equipment failures, demand fluctuations or supply chain interruptions without the need for expensive retraining cycles. Networked AI realizes edge-cloud collaborative learning by distributing decisionmaking agents at the industrial hierarchy. Each agent interactively adapts to application-specific operational feedback, and the Regulation Layer maintains global life through the regulated update rules of Eq. (6) Target production and implement security constraints.

Dynamic scheduling optimization exemplifies the advantages of the networked AI paradigm: role-based agent communities representing individual production cells or operators who negotiate resource allocation or operational flows through the Interaction Layer, guided by application-specific priors encoding process physics and quality specifications. This selforganizing behavior reduces dependence on centralized planners, and enables graceful degradation under partial system failures, mimicking exactly how human organization operates in vertical industries. As no existing industry can offer the massive high-quality data as required by large standalone models, Networked AI offers the industries an applicationspecific, intelligent and autonomous system that can be not only developed upon scarce and private industrial data, but also continuously optimized for target scenarios during its operation.

Moreover, by transforming industrial AI from a black-box optimizer into an interpretable distributed decision network, Networked AI improves both operational efficiency and system accountability, providing operators with traceable, layer-wise explanations of scheduling decisions and anomaly responses consistent with the gray-box architecture, as established in Section III-E.

## E. Cellular Networks and Beyond

Cellular networks are one of the most specific industrial settings for Networked AI, because they are already composed of a large number of distributed entities, which must coordinate under changing radio, traffic and service conditions. Such networks are built on heterogeneous infrastructure with emerging and legacy components, while serving non-fixed channels, uneven and sudden needs, mobile users, and strict latency and reliability goals. In this case, it is difficult to keep the information obtained through offline training or centralized adjustment up to date after deployment. Base stations, user equipment (UE), edge computing platforms, core-network analytics functions such as the Network Data Analytics Function (NWDAF), and operations, administration, and maintenance (OAM) systems can each be regarded as an operational agent that observes a local state and acts under network-wide service and operational policies. These entities form the Agent Layer, and recent 5G-Advanced studies on AI/ML for the NR air interface and on NG-RAN intelligence provide an initial standardization basis for this evolution [94], [95].

Cellular AI spans radio-level optimization and networklevel automation. Learning-based methods have been studied for channel state information feedback, channel prediction, beam management, positioning, and related air-interface functions [94], [96]. AI can also support traffic forecasting, energysaving control, mobility and load balancing, fault diagnosis, and slice or service assurance [97], [98]. The role of the Interaction Layer organizes telemetry, radio measurements, service indicators, and model information across the radio access network, core, edge, and terminals. When the difference between the predicted network behavior and the measured observed value exceeds the threshold, for example, ||yˆ<sub>i</sub>(t) − y $\cdot ( t , G ( t ) ) | | > \tau$ , selective local adaptation can be triggered, and only the relevant agent or service area can be updated.

Cellular infrastructure can support edge inference, federated learning, model distribution, and cloud–edge–device coordination [99], [100]. This is meaningful only when AI services are delivered with predictable performance, privacy protection, and manageable cost. The Regulation Layer addresses this requirement through service-level assurance, slice policy enforcement, safety validation, anomaly isolation, fallback, and rollback before AI decisions or model updates are widely applied. In this process, regulation operators such as $\Gamma _ { i } ( t )$ can constrain local updates under global stability and service requirements.

In addition to terrestrial cellular networks, Orbital AI is emerging, and AI workloads are directly deployed in the skybased computing infrastructure, especially the Low Earth Orbit (LEO) satellite constellation. The satellite can perform local inference, intermediate feature extraction, task unloading and model coordination across satellite links, instead of continuously downlinking the original earth observation, environment or communication data to the ground cloud. This will convert the LEO giant constellation from communication relay to a distributed orbital computing platform in the 6G era [101], [102]. The future Orbital AI system can further integrate the federal world arrangement. In such a framework, LEO, MEO (Medium Earth Orbit, commonly 5,000–20,000 km above Earth), GEO (Geostationary Earth Orbit at about 35,786 km above the equator) and ground cloud resources cooperate as a unified computing continuum. Recent work on AI-empowered non-terrestrial networks, in-orbit machine learning payloads, and satellite edge intelligence suggests that the architecture of Networked AI will become central to future 6G infrastructure [102]–[104]. Orbital AI in space is therefore not merely AI deployed on satellites; it is an adaptive, radiationaware, and energy-conscious computing paradigm designed for autonomous Networked AI beyond Earth.

## VI. INDUSTRIAL IMPACTS AND OPEN CHALLENGES

## A. Impact on 6G and Applications

Networked AI offers an organizing framework for 6G systems, in which the network is expected to evolve from a connectivity infrastructure into one that also coordinates distributed AI functions across devices, base stations, edge clouds, core-network functions, and application platforms. The IMT-2030 framework identifies “AI and Communication” as one of its usage scenarios and treats ubiquitous intelligence as an overarching aspect of future systems [105]. As a preceding effort, the NSAI research community also explicitly identified the roadmap of ”AI as a Networking System, and Network as an AI system”, which virtually equals network and AI [16]. These directions map naturally onto the threelayer Networked AI architecture for potentially universal 6G applications: heterogeneous models, edge and terminals, network functions, and intelligent application/service functions constitute the Agent Layer; cross-domain data, context, and model information are exchanged and optimized through the Interaction Layer; and validation, policy enforcement, fallback, rollback, and accountability are handled by the Regulation Layer. For 6G and its applications, the question is not whether AI can be introduced, but how AI functions can be deployed, updated, and governed at network scale.

Networked AI has also transformed several research topics into specific design problems of industrial standardization and deployment. The integration of perception, communication, computing and AI requires agent to combine multi-modal observation with distributed decision-making, while maintaining a controlled state of data and model life cycles. Energy efficiency and sustainability may benefit from online evolutive learning, because in principle, model updates can be limited to units, equipment or service areas where traffic volume, radio conditions or tasks actually change, thus avoiding networkwide retraining. When local learning, cross-domain interaction and system-level regulation are clearly separated, credibility and security problems can be solved more systematically. Network digital twins can also be used as a controlled environment to verify and stress-testing AI policies before they are applied to live networks [106].

In summary, from the perspectives of Information and Communication Technology (ICT) industry, we observe a ramping-up momentum that 6G is being built by AI and for AI. Particular for 6G applications, the value of Networked AI lies less in adding AI algorithms to individual devices than in providing a managed infrastructure for intelligent services. Typical vertical applications, including intelligent IoT, embodied, spatial, and industrial intelligence, require more than connectivity. They depend on coordinated intelligence among devices, edge nodes, network functions, and application platforms, supported by predictable latency, trusted data exchange, and accountable decision-making [99]. The impact converges with the aforementioned NSAI paradigm: AI is expected to become a native capability for operating and optimizing 6G networks, while 6G provides a deployable and governed infrastructure for AI services at industrial scale. And therefore, 6G applications are virtually Networked-AI applications, which in-turn are expected to drive the evolution of the entire ICT industry to its next generation.

## B. Key Open Challenges

Despite its potential to enable scalable and adaptive distributed intelligence, Networked AI introduces several fundamental open challenges spanning the agent, interaction, and regulation layers. These challenges arise from the need to jointly support heterogeneous intelligence, decentralized evolution, and system-level control in dynamic environments.

At the Agent Layer, one of the main challenges is to develop heterogeneous intelligent agents that can continuously adapt under resource constraints. Unlike the traditional distributed learning paradigm, Networked AI must support agents with different architectures, modalities and learning paradigms. This raises important theoretical questions about the compatibility, the transferability of learned knowledge, and the convergence under heterogeneous local objectives. The difficulties further introduced in a nonstationary environment may include catastrophic forgetting. From an application perspective, edge devices, autonomous robots, wearable systems and smart sensing platforms require lightweight yet reliable intelligence to run under limited computation, energy and communication budgets. Designing and ensuring the robustness, interpretability, privacy preservation and trustworthy model of agentic AI community is still an open research direction.

The Interaction Layer raises other challenges related to decentralized coordination and collective evolution. A central problem is to design adaptive interactive topologies that can evolve according to task requirements, environmental changes and resource availability. The existing distributed optimization and consistency framework is usually not suitable for highly dynamic and heterogeneous systems characterized by asynchronous updates, communication delays and partial participation. Another challenge involves balancing local specialization and global consistency. Excessive synchronization will inhibit diversity and adaptability, while insufficient coordination will lead to instability and fragmentation of learning behavior. In addition, understanding the emerging collective intelligence has not been explored to a large extent, especially in systems where interaction structures and proxy behaviors have evolved over time. Robustness against adversarial agents, faulty communications, and cascading error propagation is also critical for large-scale deployments.

In the Regulation Layer, the main challenge is to achieve system-level stability, controllability and reliability while maintaining decentralized autonomy. Networked AI requires a hierarchical control mechanism, which can regulate a large number of adaptive agents through global goals, constraints and feedback signals. Providing strict theoretical guarantees for the convergence, stability and safety of continuously evolving distributed systems is still an open issue. In addition, dynamic resource arrangement across computing, communication and energy dimensions is crucial for scalable operations. Regulatory mechanisms must also address ethical and societal concerns, including fairness, accountability, transparency and governance of autonomous collective behavior. These problems have become particularly important in critical applications such as health care, smart grids, autonomous transportation, and national or industrial-scale cyber-physical systems.

Beyond the individual layers, several cross-layer challenges remain unresolved. A unified theoretical framework capable of jointly modeling local intelligence, interaction dynamics, and global regulation is still at its early stage. Scalability to massive networks with millions of adaptive agents, resilience against failures and attacks, and interoperability across heterogeneous platforms remains to be proven by researchers in both academia and industry. Ultimately, the central challenge of Networked AI is to reconcile four competing objectives simultaneously: local autonomy, collective intelligence, continual evolution, and globally regulated stability. Addressing these challenges will refine the future theoretical foundations and real-world applicability of Networked AI.

## VII. CONCLUSION

Networked AI offers the potential for substantially improved data and intelligence efficiency through localized interaction, hierarchical factorization, and system-level regulation, and the foreseeable and explainable AI that can be working reliably with mankind. In this paper, we have established a threelayer hierarchical architecture of networked AI, together with a unified mathematical formalism encompassing agent state dynamics, interaction-driven signal processing, and global stability control, where the three fundamental transitions are highlighted.

Training large models with small data: near-linear vs. exponential growth of resource demand. Conventional standalone AI systems depend on massive offline datasets whose required volume can grow exponentially with system capability or task complexity. Networked AI replaces this with continuously generated, interaction-driven high-quality data in near-linear scaling. This transition shifts AI systems from data-hungry monolithic or offline training towards sustainable collective evolution, representing a fundamental improvement in data efficiency.

From generalization to application-specific adaptation: online evolution vs. offline training. Traditional models are trained once and deployed statically, struggling under distributional shift and out-of-distribution novelties. Networked AI’s Interaction Layer enables online detection of statistical discrepancies through adaptive signal processing, triggering targeted, localized application customization, i.e., online evolution, without global retraining. Smaller and agile models are therefore possible, representing a fundamental improvement in intelligence efficiency .

Achieving interpretability through networking: gray-box vs. black-box. Deep learning’s black-box nature limits controllability and trust, whereas networked AI transforms this by structuring intelligence as explicitly defined layers with explainable interfaces: signal processing kernels ground the Interaction Layer in provable estimation theory, and Lyapunov-based control provides formal stability conditions for the Regulation Layer. This layered decomposition renders the system an interpretable and amenable gray-box without sacrificing the representational power of deep models.

New quantitative metrics of data efficiency, intelligence efficiency, and interpretability remain to be discussed and defined in the research and technical communities. Nevertheless, these three transitions together are defining Networked AI not only as an engineering approach with prolific prospects, but also as a paradigm shift, transforming from isolated, data-saturated models toward collectively intelligent, resource-efficient, and interpretable systems, autonomously evolving for the target application scenarios. This transformative paradigm is being seamlessly bridged with the rapidly evolving landscape of large-scale agentic and self-improving AI systems.

## APPENDIX A SCALING LAWS AND REMARKS

Let N be the number of heterogeneous task components, $c > 1$ the feature-expansion factor, and R(N) and C(N) the sample and computational requirements for reaching a prescribed performance criterion.

Theorem 1 (Standalone scaling under global coupling). Consider a standalone end-to-end system that jointly models N heterogeneous task components. Suppose that: (i) each component contributes at least c distinguishable feature configurations; (ii) no structural factorization excludes crosscomponent combinations; and (iii) resolving each relevant joint configuration requires at least a constant number of samples and operations. Then

$$
R _ { \mathrm { s } } ( N ) = \Omega \left( c ^ { N } \right) , \qquad C _ { \mathrm { s } } ( N ) = \Omega \left( c ^ { N } \right) .\tag{19}
$$

Proof. Let $\mathcal { X } _ { i }$ be the feature-configuration set of component i, where $\left| { \mathcal { X } } _ { i } \right| \geq c .$ Without structural factorization, the joint configuration space is

$$
\mathcal { X } _ { \mathrm { s } } = \prod _ { i = 1 } ^ { N } \mathcal { X } _ { i } , \qquad | \mathcal { X } _ { \mathrm { s } } | = \prod _ { i = 1 } ^ { N } | \mathcal { X } _ { i } | \geq c ^ { N } .\tag{20}
$$

If every relevant configuration requires at least $q _ { \mathrm { s } } > 0$ samples and $r _ { \mathrm { s } } > 0$ operations per sample, then

$$
R _ { \mathrm { s } } ( N ) \geq q _ { \mathrm { s } } c ^ { N } , \qquad C _ { \mathrm { s } } ( N ) \geq r _ { \mathrm { s } } R _ { \mathrm { s } } ( N ) .\tag{21}
$$

Therefore, $R _ { \mathrm { s } } ( N ) , C _ { \mathrm { s } } ( N ) = \Omega ( c ^ { N } )$

□

Theorem 2 (Networked-AI scaling under hierarchical factorization). Consider N agents organized into a balanced hierarchy with branching factor $b > 1$ and depth

$$
\begin{array} { r } { L = \left\lceil \log _ { b } N \right\rceil . } \end{array}\tag{22}
$$

Suppose that: (i) every hierarchical level increases the featurecombination complexity along an agent-centered path by at most a factor $c ; ~ ( i i )$ interaction and optimization remain localized, avoiding an unrestricted Cartesian product across different paths; and (iii) the samples and operations per locally processed feature combination are bounded by constants independent of N. Then

$$
\begin{array} { r } { R _ { \mathrm { n } } ( N ) , C _ { \mathrm { n } } ( N ) = O \Big ( N c ^ { \lceil \log _ { b } N \rceil } \Big ) } \\ { = O \big ( N ^ { 1 + \log _ { b } c } \big ) . } \end{array}\tag{23}
$$

Proof. Along any agent-centered path, at most $c ^ { L }$ feature combinations are generated after L hierarchical levels. With at most N such paths and no unrestricted cross-path expansion, the aggregate number of locally processed feature combinations satisfies

$$
B _ { \mathrm { n } } ( N ) \le K _ { 0 } N c ^ { L }\tag{24}
$$

for some constant $K _ { 0 } > 0$ independent of N. If each locally processed combination requires at most $q _ { \mathrm { n } } > 0$ samples and $r _ { \mathrm { n } } > 0$ operations, then

$$
R _ { \mathrm { n } } ( N ) \leq q _ { \mathrm { n } } B _ { \mathrm { n } } ( N ) , \qquad C _ { \mathrm { n } } ( N ) \leq r _ { \mathrm { n } } B _ { \mathrm { n } } ( N ) .\tag{25}
$$

Hence,

$$
R _ { \mathrm { n } } ( N ) , C _ { \mathrm { n } } ( N ) = O \big ( N c ^ { L } \big ) .\tag{26}
$$

Moreover,

$$
c ^ { \lceil \log _ { b } N \rceil } \leq c ^ { 1 + \log _ { b } N } = c N ^ { \log _ { b } c } .\tag{27}
$$

Because c is independent of $N$ , substituting $L = \lceil \log _ { b } N \rceil$ into Eq. (26) yields Eq. (23). □

Corollary 1 (Scaling separation). Under the assumptions of Theorems 1 and 2,

$$
\frac { R _ { \mathrm { s } } ( N ) } { R _ { \mathrm { n } } ( N ) } , \frac { C _ { \mathrm { s } } ( N ) } { C _ { \mathrm { n } } ( N ) } = \Omega \bigg ( \frac { c ^ { N } } { N c ^ { \lceil \log _ { b } N \rceil } } \bigg ) .\tag{28}
$$

Thus, hierarchical factorization changes the resource $d e \mathrm { - }$ pendence on N from exponential under unrestricted global coupling to $O \big ( \dot { N } ^ { 1 + \mathrm { l o g } _ { b } c } \big )$ . When $\log _ { b } c \ll 1$ , the exponent approaches one; we refer to this regime as near-linear scaling.

Remark 1 (Scope). Theorem 1 is conditional on unrestricted global coupling and is not a universal lower bound for all monolithic architectures. Theorem 2 is an upper bound under localized hierarchical factorization. In particular, $c < b$ gives a subquadratic bound, whereas near-linear scaling requires the stronger regime $\log _ { b } c \ll 1$

Remark 2 (Connection to the experiments). Figure 2c evaluates $N \in \{ 2 , 3 , 4 , 5 , 6 \}$ . Over this finite range, $N ^ { 1 + \log _ { b } c }$ can become $c \cdot N$ when b is configured to N. The observed linear fit is therefore consistent with Theorem 2. Figures 2a and 2b provide complementary evidence of reduced empirical resource sensitivity with respect to model and application scales, as related to $c ;$ they do not directly show the bounds with respect to N.

## APPENDIX B

## CONDITIONAL STABILITY OF THE REGULATION

Let $g _ { i } ( t ) = \nabla _ { \Theta _ { i } } J _ { i } ( \mathcal { D } _ { i } ^ { * } ( t ) , \Theta _ { i } ( t ) )$ and $a _ { i } ( t ) = \eta \Gamma _ { i } ( t )$ for notation simplicity here, so that Eq. (11) can be written as $\Theta _ { i } ( t + 1 ) = \Theta _ { i } ( t ) - a _ { i } ( t ) g _ { i } ( t )$ . We represent the approximation in Eq. (9) as

$$
V ( S ( t ) ) = \sum _ { i = 1 } ^ { K } V _ { i } ( \Theta _ { i } ( t ) , { \mathcal { Z } } _ { i } ( t ) ) + \varepsilon ( t ) ,\tag{29}
$$

where $\varepsilon ( t )$ collects cross-agent coupling and decomposition residuals. For the stability analysis, the non-parameter components of $s _ { i } ( t )$ are absorbed into the residual $\varepsilon ( t )$

Assumption 1: The global energy function V is lower bounded, and each $V _ { i }$ has an L<sub>i</sub>-Lipschitz gradient. Moreover, for some $\mu _ { i } > 0 , \overline { { \Delta } } _ { i } < \infty$ , and $\rho _ { i } \geq 0$

$$
\langle \nabla _ { \Theta _ { i } } V _ { i } ( t ) , g _ { i } ( t ) \rangle \geq \mu _ { i } \Vert g _ { i } ( t ) \Vert ^ { 2 } ,\tag{30}
$$

$$
\Delta V _ { i } ( t ) \leq \overline { { \Delta } } _ { i } ,\tag{31}
$$

$$
\varepsilon ( t + 1 ) - \varepsilon ( t ) \leq \sum _ { i = 1 } ^ { K } \rho _ { i } a _ { i } ( t ) \| g _ { i } ( t ) \| ^ { 2 } ,\tag{32}
$$

where $0 < \eta <$ min<sub>i</sub> $2 \mu _ { i } / L _ { i }$ and $\rho _ { i } < \mu _ { i } - L _ { i } \eta / 2$

These assumed conditions are practically plausible in controlled Networked-AI deployments rather than being automatic properties of arbitrary networks. Local smoothness is commonly satisfied, or can be enforced locally, when agent objectives use smooth losses and bounded parameter updates. The first inequality Eq. (30), the descent-alignment condition, can be promoted by augmenting each task objective with a stability regularizer derived from $V _ { i } ,$ or by projecting task gradients onto a descent-compatible direction. The second inequality Eq. (31), the bounded-variation condition is reasonable when inputs, gradients, and candidate updates are clipped or otherwise constrained, which prevents $\Gamma _ { i } ( t )$ from vanishing. The third inequality Eq. (32), the coupling-residual bound is compatible with the localized architecture when neighborhood sizes, interaction weights, communication delays, and environmental drift remain bounded. The learning-rate and residualmargin conditions afterward can then be checked or enforced online by conservative step-size selection and monitoring of the estimated descent margin.

Theorem 3 (Conditional Stability). Under Assumption 1, the regulated dynamics satisfy

$$
V ( S ( t + 1 ) ) - V ( S ( t ) ) \leq - \sum _ { i = 1 } ^ { K } a _ { i } ( t ) c _ { i } \| g _ { i } ( t ) \| ^ { 2 } \leq 0 ,\tag{33}
$$

where $c _ { i } = \mu _ { i } - L _ { i } \eta / 2 - \rho _ { i } > 0 .$ . Consequently, $V ( S ( t ) )$ converges and $\| g _ { i } ( t ) \|  0 f o r$ every agent. If the resulting invariant stationary set contains a unique equilibrium $S ^ { \star }$ and V is positive definite with respect to $S ^ { \star }$ , then $S ^ { \star }$ is asymptotically stable.

Proof. By the descent lemma and Assumption 1,

$$
V _ { i } ( t + 1 ) - V _ { i } ( t ) \leq - a _ { i } ( t ) \langle \nabla _ { \Theta _ { i } } V _ { i } ( t ) , g _ { i } ( t ) \rangle + \frac { L _ { i } } { 2 } a _ { i } ^ { 2 } ( t ) \| g _ { i } ( t ) \| ^ { 2 }\tag{34}
$$

$$
\leq - a _ { i } ( t ) \left( \mu _ { i } - \frac { L _ { i } \eta } { 2 } \right) \| g _ { i } ( t ) \| ^ { 2 } ,\tag{35}
$$

because $0 ~ < ~ a _ { i } ( t ) ~ \le ~ \eta .$ . Summing over i and applying the residual bound gives the stated descent inequality $\operatorname { E q } .$ (33). Since Eq. (10) and $\Delta V _ { i } ( t ) ~ \le ~ \overline { { \Delta } } _ { i }$ imply $a _ { i } ( t ) \geq$ $\eta \exp ( - \kappa \overline { { \Delta } } _ { i } ) > 0 .$ , summing the descent inequality over time yields $\| g _ { i } ( t ) \| \to 0$ . The final stability claim follows under the additional conditions stated above. □

## REFERENCES

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

[4] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving language understanding by generative pre-training,” [Online]. Available: https://cdn.openai.com/research-covers/ language-unsupervised/language understanding paper.pdf, 2018.

[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” 2021. [Online]. Available: https: //arxiv.org/abs/2103.00020

[6] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 8821–8831. [Online]. Available: https://proceedings.mlr.press/v139/ramesh21a.html

[7] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, and et al., “Language models are few-shot learners,” 2020.

[8] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, and et al., “Palm: Scaling language modeling with pathways,” 2022. [Online]. Available: https://arxiv.org/abs/2204.02311

[9] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[10] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” 2020. [Online]. Available: https://arxiv.org/abs/2001.08361

[11] R. Sutton, “The bitter lesson,” [Online]. Available: https://www.cs. utexas.edu/<sup>∼</sup>eunsol/courses/data/bitter lesson.pdf, 2019.

[12] U. Sobal, W. Zhang, K. Cho, R. Balestriero, T. G. J. Rudner, and Y. LeCun, “Learning from reward-free offline data: A case for planning with latent dynamics models,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 43 905–43 941.

[13] R. Goswami, P. Krishnamurthy, Y. LeCun, and F. Khorrami, “OSVI-WM: One-shot visual imitation for unseen tasks using world-modelguided trajectory generation,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 54 725–54 745.

[14] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 10 632–10 643.

[15] Y. Bengio, M. Cohen, D. Fornasiere, J. Ghosn, P. Greiner, M. MacDermott, S. Mindermann, A. Oberman, J. Richardson, O. Richardson, M.-A. Rondeau, P.-L. St-Charles, and D. Williams-King, “Superintelligent agents pose catastrophic risks: Can scientist ai offer a safer path?” 2025. [Online]. Available: https://arxiv.org/abs/ 2502.15657

[16] L. Song, X. Hu, G. Zhang, P. Spachos, K. Plataniotis, and H. Wu, “Networking systems of AI: on the convergence of computing and communications,” IEEE Internet of Things Journal, vol. 9, no. 20, pp. 20 352–20 381, 2022.

[17] L. Qian, P. Sun, Y. Wang, J. Jin, A. Boukerche, and L. Song, “A new online evolutive optimization method for driving agents,” in GLOBECOM 2024-2024 IEEE Global Communications Conference. IEEE, 2024, pp. 2244–2249.

[18] L. Qian, J. Jin, P. Sun, J. Liu, B. Hu, A. Boukerche, and L. Song, “A coordinated optimization framework for intelligent agents with online evolutive learning,” IEEE Internet of Things Journal, vol. 13, no. 11, pp. 23 172–23 189, 2026.

[19] Y. Bengio, S. Clare, C. Prunkl, and et al., “International AI safety report 2026,” 2026. [Online]. Available: https://arxiv.org/abs/2602.21012

[20] Y. Bengio, G. Hinton, A. Yao, D. Song, P. Abbeel, T. Darrell, Y. N. Harari, Y.-Q. Zhang, L. Xue, S. Shalev-Shwartz, G. Hadfield, J. Clune, T. Maharaj, F. Hutter, A. G. Baydin, S. McIlraith, Q. Gao, A. Acharya, D. Krueger, A. Dragan, P. Torr, S. Russell, D. Kahneman, J. Brauner, and S. Mindermann, “Managing extreme AI risks amid rapid progress,” Science, vol. 384, no. 6698, pp. 842–845, 2024.

[21] J. Jin, L. Qian, and L. Song, “Multi-agent framework based on coordinated sensing and control with online evolutive learning,” in 2024 IEEE 10th World Forum on Internet of Things (WF-IoT). IEEE, 2024.

[22] D. Li, X. Zhu, and L. Song, “Mutual match for semi-supervised online evolutive learning,” Applied Intelligence, vol. 53, pp. 3336–3350, 2023.

[23] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[24] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[25] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[26] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.

[27] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.

[28] OpenAI, “Hello GPT-4o,” https://openai.com/index/hello-gpt-4o, 2024, accessed: 2026-04-14.

[29] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[30] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[31] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π<sub>0</sub>: A visionlanguage-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[32] X. Zhou, X. Han, F. Yang, Y. Ma, V. Tresp, and A. Knoll, “Opendrivevla: Towards end-to-end autonomous driving with large vision language action model,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 16, 2026, pp. 13 782–13 790.

[33] W. Zuo, Y. Liu, W. Yang, F. Wu, W. Zhou, J. Liu, P. Sun, J. Cheng, D. Yang, and X. Zeng, “Embodied multimedia: When multimedia meets embodied intelligence,” Available at SSRN 6456578, 2026.

[34] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in The eleventh international conference on learning representations, 2022.

[35] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” Advances in neural information processing systems, vol. 36, pp. 68 539–68 551, 2023.

[36] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu et al., “Autogen: Enabling next-gen LLM applications via multi-agent conversations,” in First conference on language modeling, 2024.

[37] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[38] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[39] Y. Liu, J. Liu, K. Yang, B. Ju, S. Liu, Y. Wang, D. Yang, P. Sun, and L. Song, “Amp-net: Appearance-motion prototype network assisted automatic video anomaly detection system,” IEEE Transactions on Industrial Informatics, vol. 20, no. 2, pp. 2843–2855, 2023.

[40] B. Chen, L. Cao, W. Liu, J. Liu, B. Zhao, and Y. Liu, “Crossmodal attention fusion of rgb and skeleton for multimodal-driven video anomaly detection,” Pattern Recognition, p. 113815, 2026.

[41] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[42] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[43] M. L. Puterman, Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[44] R. S. Sutton, A. G. Barto et al., Reinforcement learning: An introduction. MIT press Cambridge, 1998, vol. 1, no. 1.

[45] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, and Q. V. Le, “MnasNet: Platform-aware neural architecture search for mobile,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 2820–2828.

[46] J. Lin, W.-M. Chen, Y. Lin, C. Gan, S. Han et al., “MCUNet: Tiny deep learning on iot devices,” Advances in neural information processing systems, vol. 33, pp. 11 711–11 722, 2020.

[47] R. E. Kalman and R. S. Bucy, “New results in linear filtering and prediction theory,” 1961.

[48] N. Wiener, Extrapolation, interpolation, and smoothing of stationary time series: with engineering applications. The MIT press, 1949.

[49] R. L. Plackett, “Some theorems in least squares,” Biometrika, vol. 37, no. 1/2, pp. 149–157, 1950.

[50] D. I. Shuman, S. K. Narang, P. Frossard, A. Ortega, and P. Vandergheynst, “The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains,” IEEE signal processing magazine, vol. 30, no. 3, pp. 83–98, 2013.

[51] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[52] P. Pulkkinen and V. Koivunen, “Model-based online learning for active isac waveform optimization,” IEEE Journal ofSelected Topics in Signal Processing, vol. 18, no. 5, pp. 737–751, 2024.

[53] A. M. Lyapunov, “The general problem of the stability of motion,” International journal of control, vol. 55, no. 3, pp. 531–534, 1992.

[54] J. Richalet, A. Rault, J.-L. Testud, and J. Papon, “Model predictive heuristic control,” Automatica (journal of IFAC), vol. 14, no. 5, pp. 413–428, 1978.

[55] J. Jin, L. Qian, J. Cai, P. Sun, and L. Song, “Achieving linear scalability in networked ai: a coordinated sensing and control approach,” in 2025 IEEE 11th World Forum on Internet of Things (WF-IoT). IEEE, 2025, pp. 1–6.

[56] P. Zambare, V. N. Thanikella, and Y. Liu, “Securing agentic AI: Threat modeling and risk analysis for network monitoring agentic ai system,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10043

[57] Z. Deng, Y. Guo, C. Han, W. Ma, J. Xiong, S. Wen, and Y. Xiang, “AI agents under threat: A survey of key security challenges and future pathways,” ACM Computing Surveys, vol. 57, no. 7, Jul. 2025. [Online]. Available: https://doi.org/10.1145/3716628

[58] S. Dong, S. Xu, P. He, Y. Li, J. Tang, T. Liu, H. Liu, and Z. Xiang, “Memory injection attacks on LLM agents via query-only interaction,” in Advances in Neural Information Processing Systems

38, 2025, also available as arXiv:2503.03704. [Online]. Available: https://neurips.cc/virtual/2025/poster/118152

[59] E. Debenedetti, I. Shumailov, T. Fan, J. Hayes, N. Carlini, D. Fabian, C. Kern, C. Shi, A. Terzis, and F. Tramer, “Defeating prompt injections\` by design,” 2025. [Online]. Available: https://arxiv.org/abs/2503.18813

[60] K. Huang, V. S. Narajala, J. Yeoh, J. Ross, R. Raskar, Y. Harkati, J. Huang, I. Habler, and C. Hughes, “A novel zero-trust identity framework for agentic AI: Decentralized authentication and finegrained access control,” 2025. [Online]. Available: https://arxiv.org/ abs/2505.19301

[61] P. Sun, X. Liu, Z. Wang, and B. Liu, “Byzantine-robust decentralized federated learning via dual-domain clustering and trust bootstrapping,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 24 756–24 765.

[62] Y. Louck, A. Stulman, and A. Dvir, “Improving google A2A protocol: Protecting sensitive data and mitigating unintended harms in multi-agent systems,” ACM Transactions on Software Engineering and Methodology, Jun. 2026. [Online]. Available: https://doi.org/10. 1145/3821216

[63] ——, “Security analysis of agentic AI communication protocols: A comparative evaluation,” 2025. [Online]. Available: https://arxiv.org/ abs/2511.03841

[64] Q. Zhan, R. Fang, H. S. Panchal, and D. Kang, “Adaptive attacks break defenses against indirect prompt injection attacks on LLM agents,” in Findings of the Association for Computational Linguistics: NAACL 2025. Albuquerque, New Mexico: Association for Computational Linguistics, Apr. 2025, pp. 7116–7132. [Online]. Available: https://aclanthology.org/2025.findings-naacl.395/

[65] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann et al., “Palm: Scaling language modeling with pathways,” Journal of Machine Learning Research, vol. 24, no. 240, pp. 1–113, 2023.

[66] M. Wooldridge, An Introduction to MultiAgent Systems, 2nd ed. John Wiley & Sons, 2009.

[67] A. Dorri, S. S. Kanhere, and R. Jurdak, “Multi-agent systems: A survey,” IEEE Access, vol. 6, pp. 28 573–28 593, 2018.

[68] Y. Cao, W. Yu, W. Ren, and G. Chen, “An overview of recent progress in the study of distributed multi-agent coordination,” IEEE Transactions on Industrial Informatics, vol. 9, no. 1, pp. 427–438, 2013.

[69] K. Zhang, Z. Yang, H. Liu, T. Zhang, and T. Bas¸ar, “Fully decentralized multi-agent reinforcement learning with networked agents,” in International Conference on Machine Learning (ICML). PMLR, 2018, pp. 5872–5881.

[70] L. Bus¸oniu, R. Babuska, and B. De Schutter, “A comprehensive surveyˇ of multiagent reinforcement learning,” IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 38, no. 2, pp. 156–172, 2008.

[71] K. Zhang, Z. Yang, and T. Bas¸ar, “Multi-agent deep reinforcement learning: A survey,” Artificial Intelligence Review, vol. 54, no. 6, pp. 3959–3996, 2021.

[72] B. McMahan, E. Moore, D. p. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial Intelligence and Statistics (AISTATS). PMLR, 2017, pp. 1273–1282.

[73] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, 2020.

[74] Y. Liu, S. Liu, X. Zhu, J. Li, H. Yang, L. Teng, J. Guo, Y. Wang, D. Yang, and J. Liu, “Privacy-preserving video anomaly detection: A survey,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–22, 2025.

[75] M. Ribero, H. Vikalo, and G. de Veciana, “Federated learning at scale: Addressing client intermittency and resource constraints,” IEEE Journal of Selected Topics in Signal Processing, vol. 19, no. 1, pp. 60–73, 2024.

[76] A. Lalitha, S. Shekhar, T. Javidi, and F. Koushanfar, “Peer-to-peer federated learning on graphs,” arXiv preprint arXiv:1901.11173, 2019.

[77] T. Sun, D. Li, and B. Wang, “Decentralized federated averaging,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4289–4301, 2022.

[78] A. Z. Tan et al., “Personalized federated learning via heterogeneous modular networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[79] J. Xu, S. Wan, Y. Li, S. Luo, Z. Chen, Y. Shao, Z. Chen, S.-L. Huang, and L. Song, “Cooperative multi-model training for personalized federated learning over heterogeneous devices,” IEEE Journal of Selected Topics in Signal Processing, vol. 19, no. 1, pp. 195–207, 2024.

[80] Y. Zhu et al., “Federated multi-agent reinforcement learning: A comprehensive survey,” Expert Systems with Applications, 2024.

[81] S. Wagle, A. B. Das, D. J. Love, and C. G. Brinton, “Multi-agent reinforcement learning for graph discovery in d2d-enabled federated learning,” IEEE Journal of Selected Topics in Signal Processing, 2025.

[82] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Networks, vol. 113, pp. 54–71, 2019.

[83] M. Mundt, Y. W. Hong, I. Pliushch, and V. Ramesh, “A wholistic view of continual learning with deep neural networks,” Neural Networks, vol. 160, pp. 306–336, 2023.

[84] Z. Chen and B. Liu, Lifelong Machine Learning, 2nd ed. Morgan & Claypool Publishers, 2018.

[85] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, “Towards continual reinforcement learning: A review and perspectives,” Journal of Artificial Intelligence Research (JAIR), vol. 75, pp. 1401–1476, 2022.

[86] Y. Liu, H. Wang, Z. Wang, X. Zhu, J. Liu, P. Sun, R. Tang, J. Du, V. Leung, and L. Song, “Crcl: Causal representation consistency learning for anomaly detection in surveillance videos,” IEEE Transactions on Image Processing, vol. 34, pp. 2351–2366, 2025.

[87] A. Soltoggio et al., “A collective AI via lifelong learning and sharing at the edge,” Nature Machine Intelligence, vol. 6, pp. 251–264, 2024.

[88] E. Bonabeau, M. Dorigo, and G. Theraulaz, Swarm Intelligence: From Natural to Artificial Systems. Oxford University Press, 1999.

[89] J. Kennedy, R. Eberhart, and Y. Shi, Swarm Intelligence. Morgan Kaufmann, 2001.

[90] K. Cao, A. Lazaridou, M. Lanctot, J. Z. Leibo, K. Tuyls, and S. Clark, “Emergent communication through negotiation,” in International Conference on Learning Representations (ICLR), 2018.

[91] V. Conitzer, C. Oesterheld et al., “Foundations of cooperative AI,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 13, 2023, pp. 15 359–15 367.

[92] C. Xu, R. Qian, H. Fang, X. Ma, W. I. Atlas, J. Liu, and M. A. Spoljaric, “Salina: Towards sustainable live sonar analytics in wild ecosystems,” in Proceedings of the 22nd ACM Conference on Embedded Networked Sensor Systems, ser. SenSys ’24, 2024, p. 68–81.

[93] C. Xu, J. Li, M. Jin, W. I. Atlas, M. A. Spoljaric, E. C. Ngai, and J. Liu, “Fused: Toward federated multimodal retrieval across sovereign data domains,” in Proceedings of the ACM Web Conference 2026, ser. WWW ’26, 2026, p. 8950–8960.

[94] 3GPP, “Study on artificial intelligence (AI)/machine learning (ML) for NR air interface,” 3rd Generation Partnership Project, Technical Report TR 38.843, 2024, release 18, Version 18.0.0.

[95] X. Lin, L. Kundu, C. Dick, and S. Velayutham, “Embracing AI in 5G-Advanced toward 6G: A joint 3GPP and O-RAN perspective,” IEEE Communications Standards Magazine, vol. 7, no. 4, pp. 76–83, 2023.

[96] J. Guo, C.-K. Wen, S. Jin, and G. Y. Li, “Overview of deep learningbased CSI feedback in massive mimo systems,” IEEE Transactions on Communications, vol. 70, no. 12, pp. 8017–8045, 2022.

[97] 3GPP, “Study on enhancement for data collection for NR and EN-DC,” 3rd Generation Partnership Project, Technical Report TR 37.817, 2022, release 17, Version 17.0.0.

[98] E. Coronado, R. Behravesh, T. Subramanya, A. Fernandez-Fernandez, M. S. Siddiqui, X. Costa-Perez, and R. Riggio, “Zero touch management: A survey of network automation solutions for 5G and 6G networks,” IEEE Communications Surveys & Tutorials, vol. 24, no. 4, pp. 2535–2578, 2022.

[99] K. B. Letaief, Y. Shi, J. Lu, and J. Lu, “Edge artificial intelligence for 6G: Vision, enabling technologies, and applications,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 5–36, 2022.

[100] Z. Ji and Z. Qin, “Computational offloading in semantic-aware cloudedge-end collaborative networks,” IEEE Journal of Selected Topics in Signal Processing, vol. 18, no. 7, pp. 1235–1248, 2024.

[101] H. Zhao, L. Chen, Y. C. Chou, H. Fang, and J. Liu, “[emerging ideas] orbittransit: Traffic delivery and diffusion for earth observation via satellite mobility,” in Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services, ser. MobiSys ’26, 2026, p. 202–215.

[102] S. Mahboob and L. Liu, “Revolutionizing future connectivity: A contemporary survey on AI-empowered satellite-based non-terrestrial networks in 6G,” IEEE Communications Surveys & Tutorials, vol. 26, no. 2, pp. 1279–1321, 2024.

[103] G. Mateo-Garcia, J. Veitch-Michaelis, C. Purcell, N. Longepe, S. Reid, A. Anlind, F. Bruhn, J. Parr, and P. P. Mathieu, “In-orbit demonstration of a re-trainable machine learning payload for processing optical imagery,” Scientific Reports, vol. 13, no. 1, 2023.

[104] S. Huang, L. Wang, X. Wang, B. Tan, W. Ni, and K.-K. Wong, “Edge intelligence in satellite-terrestrial networks with hybrid quantum computing,” IEEE Wireless Communications Letters, vol. 14, no. 5, pp. 1341–1345, 2025.

[105] ITU-R, “Framework and overall objectives of the future development of imt for 2030 and beyond,” International Telecommunication Union Radiocommunication Sector, Recommendation ITU-R M.2160-0, 2023.

[106] Z. Tao, W. Xu, Y. Huang, X. Wang, and X. You, “Wireless network digital twin for 6G: Generative AI as a key enabler,” IEEE Wireless Communications, vol. 31, no. 4, pp. 24–31, 2024.