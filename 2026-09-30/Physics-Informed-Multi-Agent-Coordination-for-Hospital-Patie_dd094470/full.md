# Physics-Informed Multi-Agent Coordination for Hospital Patient Flow Optimization

Guoqing Zhang<sup>1[0009-0007-6956-0814]</sup>, Rafik Hadfi<sup>1[0000-0003-2352-1936]</sup>, and <sub>Takayuki</sub> <sub>Ito</sub>1[0000-0001-5093-3886]

Graduate School of Informatics, Kyoto University, Kyoto, Japan zhang.guoqing.27t@st.kyoto-u.ac.jp, {rafik.hadfi, ito}@i.kyoto-u.ac.jp

Abstract. Eficient patient flow coordination across autonomous hospital departments is critical for mitigating overcrowding and balancing resource utilization. While classical queueing theory, specifically open Baskett–Chandy–Muntz–Palacios (BCMP) networks, provides an interpretable mathematical topology for healthcare operations, analytical mod els rely on stationary assumptions and fixed routing matrices that degrade under state-dependent real-world dynamics. Conversely, centralized reinforcement learning approaches struggle to accommodate the decentralized structure of hospital governance, where individual clinical departments function with localized observations, heterogeneous resources, and divergent operational objectives. In this paper, we present a Multi-Agent Systems (MAS) framework titled Physics-Informed Multi-Agent Coordination, which embeds empirically calibrated BCMP queueing topologies as physical priors within a decentralized multi-agent reinforcement learning architecture. Formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) under coupled resource constraints, our method enables autonomous departmental agents to cooperatively negotiate patient routing and dynamic service scaling. To mitigate environmental non-stationarity without inducing excessive communication overhead, agents exchange localized action fingerprints along network edges and optimize a spatially decomposed reward structure. Empirical evaluations driven by real-world MIMIC-IV patient trajectories indicate that this cooperative multi-agent approach substantially reduces cumulative system delay compared to static Markovian approximations, heuristic dispatching, and independent multi-agent baselines, while maintaining clinical safety constraints.

Keywords: Multi-Agent Coordination · Decentralized POMDP · Reinforcement Learning · BCMP Queueing Networks · Healthcare Operations.

## 1 Introduction

Modern large-scale hospitals are intrinsically decentralized organizations characterized by simultaneous interactions among autonomous clinical units, such as the Emergency Department (ED), Intensive Care Units (ICUs), surgical suites, and general inpatient wards. These specialized departments continuously manage heterogeneous patient populations while competing for finite, highly shared medical resources including physical beds, clinical staf, and diagnostic equipment. As global emergency healthcare demand continuously surges, hospitals face acute pressures to safely and expediently coordinate patient movement across clinical stages [1,21]. When inter-departmental coordination breaks down, localized congestion cascades rapidly into emergency boarding, protracted waiting times, and systemic operational failure.

Optimizing hospital-wide patient flow presents a challenging multi-agent coordination problem. Existing healthcare operations research has traditionally approached this problem through centralized heuristic scheduling, mixed-integer programming, static capacity planning, or classical analytical queueing theory [2, 12–14, 17]. In particular, considerable efort has focused on decision support and matheuristics for automated patient-to-bed assignment under stochastic stay lengths [3, 8, 20, 22]. Among analytical models, open Baskett–Chandy– Muntz–Palacios (BCMP) queueing networks are frequently employed to represent multi-class patient progression, probabilistic transitions, and diverse service disciplines under steady-state conditions. However, analytical BCMP formulas assume stationary arrival rates, fixed transition probabilities, and independent node queues. In clinical practice, departments operate with finite bed capacities; when a downstream unit reaches saturation, transfers from upstream wards are blocked. Under these state-dependent routing dynamics, the mathematical assumptions required for product-form equilibrium fail, limiting the applicability of static queueing solutions for dynamic operational intervention.

From a Multi-Agent Systems (MAS) perspective, a centralized control architecture (whether driven by global mathematical heuristics or monolithic deep reinforcement learning, such as PPO or DQN) is structurally ill-suited for hospitalwide clinical scheduling. In operational reality, clinical departments operate as functionally and administratively decentralized entities governed by supervisory hospital management boards [7], characterized by local observations, local operational resources, and naturally conflicting clinical objectives. For example, the Emergency Department strives to accelerate patient discharge to prevent reception overflow; the ICU operates under strict admission thresholds to reserve highacuity ventilators for critical surgeries; and general wards aim to maintain stable bed occupancy without exhausting nursing shifts. Attempting to force these distinct domains under a centralized control algorithm creates computational bottlenecks, communication overhead, and organizational friction. Instead, patient flow coordination must be modeled as decentralized decision-making under coupled resource constraints, which represents a canonical paradigm in advanced MAS research.

The broader availability of longitudinal Electronic Health Record (EHR) datasets, such as MIMIC-IV [11], facilitates empirical, data-driven system modeling. Rather than discarding classical queueing mechanics in favor of unconstrained black-box simulators, we utilize BCMP networks to establish topological structure and trafic conservation bounds. In this paper, we formulate a Physics-Informed Multi-Agent Coordination framework. Within this architecture, the empirical BCMP network acts as a structural physical prior that defines network connectedness, clinical transition pathways, and server capacity thresholds, while cooperative multi-agent reinforcement learning (MARL) negotiates patient routing and capacity scaling across localized clinical nodes.

The core technical and empirical contributions of this work are summarized as follows:

– Physics-Informed MAS Framework: We integrate classical queueing theory with multi-agent control by leveraging empirical BCMP network dynamics calibrated on MIMIC-IV as structural physical priors. This formulation provides topology and resource capacity boundaries for multi-agent coordination in finite-server environments.

Topology-Aware Decentralized Coordination Mechanism: We formulate hospital resource management as a Dec-POMDP and implement an action-fingerprint communication protocol. By sharing smoothed policy representations along topological edges, adjacent agents mitigate partial observability and environmental non-stationarity without global communication overhead.

Safety-Constrained Cooperative Routing Policy: We formulate patient flow diversion as an emergent multi-agent routing policy driven by localized negotiations between adjacent departments. To maintain clinical protocol integrity and prevent hazardous patient routing, policy action spaces are modulated by an online expert action masking mechanism derived from historical clinician transition trajectories.

– Empirical Performance Analysis: Through clinical dataset calibrations and ablation experiments, we show that cooperative multi-agent action decomposition helps suppress non-linear congestion cascades. Our approach achieves an order-of-magnitude reduction in cumulative patient delay compared to static analytical approximations, while consistently outperforming heuristic dispatchers and independent MARL baselines.

## 2 Related Work

## 2.1 BCMP Queueing Networks in Healthcare Operations

The application of BCMP queueing networks to hospital patient flow represents a mature lineage within medical operations research. Classical works, such as Armony et al. [1], provided analytical foundations for characterizing multistage inpatient trajectories and establishing theoretical occupancy thresholds. Extending this direction, Jebbor et al. [10] utilized empirical hospital statistics to jointly optimize human stafing levels and physical bed configurations within an open queueing framework, demonstrating significant waiting time reductions upon real-world deployment. Recent eforts by Mizuno et al. [16] advanced resource allocation planning by incorporating heterogeneous service times into steady-state analytical bounds.

While analytical BCMP hospital models ofer structural clarity, they assume stationary equilibrium behavior. To maintain a tractable product-form distribution, these models require that patient transition probabilities between departments remain fixed and independent of system load. In realistic clinical settings, finite departmental server capacity induces physical blocking: when downstream diagnostic or surgical units saturate, transfer pathways close and upstream queues accumulate non-linearly. Under such state-dependent transition dynamics, Markovian product-form independence holds no longer, making stationary mathematical derivations unsuitable for real-time control intervention.

## 2.2 Multi-Agent Systems and Collaborative Control

To manage interconnected dynamic environments without relying on intractable analytical solvers, Multi-Agent Systems (MAS) have emerged as a dominant paradigm in intelligent decision-making. In domains such as trafic network signal control, logistics dispatching, and power grids, formulating resource allocation as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) allows independent actors to make localized decisions under global systemic objectives [5]. In recent years, multi-agent reinforcement learning (MARL) has gained momentum across decentralized healthcare applications, demonstrating potential in collaborative clinical decision-making [6], medical resource allocation under imperfect information [9], UAV-enabled Internet of Medical Things (IoMT) infrastructures [18,19], and emergency transportation routing [15]. However, these existing healthcare MARL architectures predominantly treat the clinical ecosystem as an abstract black-box simulator. By ignoring the fundamental queueing mechanics and physical capacity limits inherent to hospital dynamics, independent learning actors fail in tightly coupled wards due to environmental non-stationarity; when multiple departmental agents simultaneously update their routing preferences without communication, localized optimization traps and oscillating downstream bottlenecks emerge.

Recent advances in MARL emphasize structured collaborative mechanisms, such as Value-Decomposition Networks (VDN), QMIX, and Multi-Agent Actor-Critic architectures equipped with spatial communicative layers or action fingerprints [5]. While these collaborative algorithms excel in unconstrained simulation benchmarks, deploying multi-agent control directly to safety-critical healthcare networks remains largely unaddressed. Unlike games or abstract routing grids, hospital dispatching cannot tolerate unconstrained structural exploration or myopic global optimization that sacrifices individual patient pathways. Our study bridges this disconnect by demonstrating how empirical BCMP queueing topologies can function as structural domain physics, guiding multi-agent cooperation, bounding negotiation spaces, and ensuring verifiable clinical safety.

## 3 Physics Prior: BCMP Queueing Topology and Constraints

Rather than utilizing queueing derivations to calculate steady-state operational limits, we repurpose the mathematical architecture of open BCMP networks as the foundational physics engine and structural boundary condition for multiagent interaction.

## 3.1 Network Topology and Trafic Conservation

We model the physical infrastructure of a hospital as a directed topological network consisting of M discrete clinical nodes $\mathcal { N } = \{ 1 , \dots , M \}$ , representing departments such as the ED, Surgery, Intensive Care Units, and General Medicine wards. Arriving patients are stratified into clinical job classes $\mathcal { K } = \{ 1 , \ldots , K \}$ based on acuity and Diagnosis-Related Group (DRG) characteristics.

As depicted in Figure 1, external patients enter the hospital according to class-specific arrival rates $\gamma _ { i , k }$ , and navigate between departments governed by an intrinsic transition probability matrix $P = \left[ p _ { ( i , k ) , ( j , r ) } \right]$ . Under classical queueing mechanics, trafic conservation dictates that the efective macroscopic arrival rate $\lambda _ { i , k }$ at any department node i satisfies the linear equilibrium:

$$
\lambda _ { i , k } = \gamma _ { i , k } + \sum _ { j \in { \cal N } } \sum _ { r \in { \cal K } } \lambda _ { j , r } \cdot p _ { ( j , r ) , ( i , k ) } , \quad \forall i \in \mathcal { N } , k \in \mathcal { K } .\tag{1}
$$

Because every hospital patient pathway eventually culminates in clinical discharge or transfer to an external absorbing state, the spectral radius $\rho ( P )$ remains strictly less than 1, guaranteeing that Equation (1) yields a unique baseline ofered workload intensity $\begin{array} { r } { a _ { i } = \sum _ { k \in \mathcal { K } } ( \lambda _ { i , k } / \mu _ { i , k } ) } \end{array}$ across all departments, where $\mu _ { i , k }$ denotes the baseline service rate (reciprocal of expected stay duration).

## 3.2 Physical Capacity Bounds and Stability Redlines

To prevent severe queue accumulation and ensure that state transitions remain positive recurrent, classical Foster-Lyapunov stability theory dictates that each department’s total ofered trafic must not exceed its physical capacity limit $c _ { i }$ (total beds or treatment stations). Thus, the operational trafic utilization $\rho _ { i }$ must satisfy:

$$
\rho _ { i } = \frac { a _ { i } } { c _ { i } } = \frac { 1 } { c _ { i } } \sum _ { k \in \mathcal { K } } \frac { \lambda _ { i , k } } { \mu _ { i , k } } < 1 , \quad \forall i \in \mathcal { N } .\tag{2}
$$

To determine baseline structural bed capacities $c _ { i } ^ { * }$ from historical empirical data, our system incorporates a Quality-and-Eficiency-Driven (QED) capacity calculation combining iterative Erlang-C delay probability verification with the Halfin-Whitt square-root stafing rule, expressed as $c _ { i } ^ { ( 0 ) } = \operatorname* { m a x } ( 1 , \lceil a _ { i } + \beta _ { i } \sqrt { a _ { i } } \rceil )$ By setting targeted quality thresholds $( \epsilon _ { \mathrm { E D } } = 0 . 0 1 , \epsilon _ { \mathrm { w a r d } } = 0 . 0 5 )$ , the physics prior establishes realistic boundaries on hardware scaling.

(a) Macro-Level: Hospital BCMP Queueing Topology and Neighbor Communication Network  
![](images/db6987afe7651cf751f376881759394411181185ead6d04a8ccc15b71985e48e.jpg)  
Fig. 1. Physics-Informed Multi-Agent Coordination Architecture. (a) Macro-Level: The hospital BCMP queueing network calibrated on MIMIC-IV patient trajectories serves as the structural topology and resource capacity boundary. (b) Micro-Level: Closed-loop system architecture for each departmental agent controller at clinical node i. The decentralized neural actor monitors local ward telemetry o<sub>i</sub>(t) alongside neighbor action fingerprints (FP<sub>j</sub>). To relieve operational congestion, the actor governs two control variables: adjusting service rate acceleration (α-Policy) to expand bed discharge capacity, and modulating cooperative patient transfer routing (λ-Policy). Meanwhile, an expert prior action mask ensures topological safety by filtering medically unfeasible transitions from the exploration space.

## 3.3 The Breakdown of Stationary Product Forms

While Equation (1) defines theoretical demand, real hospital nodes possess finite queuing capacities. When a department experiences acute surges and bed occupancy approaches $c _ { i } ,$ physical blocking occurs. In accordance with BCMP network conventions, departments operate under First-Come, First-Served (FCFS) service discipline; when concurrent transfers exceed available capacity $c _ { i } .$ , incoming patients queue in order of their arrival timestamps, temporarily boarding in upstream units until downstream beds are vacated. Consequently, traditional stationary transition constants $p _ { ( i , k ) , ( j , r ) }$ abruptly degrade into complex, timevarying dynamic routing functions $p _ { \left( i , k \right) , \left( j , r \right) } ( { \bf n } )$ driven by the immediate global congestion vector $\mathbf { n } = \left( \mathbf { n } _ { 1 } , \ldots , \mathbf { n } _ { M } \right)$

This state-dependent routing disrupts local balance conditions, precluding the derivation of tractable analytical equilibrium formulas. In the presence of an expanding joint state space, stationary analytical models become insuficient for optimal dispatch control. Nevertheless, the structural topology of the network, specifically the directional routing graph, baseline service capacities $\mu _ { i , k }$ , and utilization limits $\rho _ { i } < 1 . 0$ , supplies an informative physical prior. We utilize this architectural domain knowledge to parameterize our decentralized multi-agent coordination system, linking static analytical bounds with active cooperative governance.

## 4 Multi-Agent Coordination Framework

To navigate the dynamic complexities of finite-capacity queueing networks, we formulate hospital patient flow management as an intelligent multi-agent collaboration problem.

## 4.1 Why Multi-Agent Over Centralized RL?

A primary structural consideration when modeling hospital congestion is why a decentralized multi-agent architecture is preferable to a monolithic centralized controller (such as a single global PPO or DQN policy). In operational institutions, centralized policy execution faces three practical constraints:

Local Observations and Privacy: Departmental supervisors observe realtime parameters locally within their respective wards (e.g., immediate queue backlogs and specialized bed utilization). Streaming raw clinical variables across all hospital units to a global processor creates unnecessary communication bandwidth costs and conflicts with administrative privacy boundaries.

– Naturally Conflicting Objectives: Departments operate with disparate operational goals. The ED seeks rapid admission evacuation to keep reception bay doors open; ICUs operate under rigorous admission thresholds to reserve mechanical ventilators for sudden surgeries; general wards strive to smooth nursing shifts and stabilize occupancy. A centralized scalar reward invariably marginalizes localized department priorities, triggering systemic instability.

Decentralized Actions Under Coupled Constraints: Resource interventions, including discharge acceleration and admission control, are managed locally by departmental directors. However, these localized decisions intersect across topological pathways, as an ICU transfer rejection forces emergency department boarding and triggers upstream congestion cascades.

Thus, hospital operational dynamics represent decentralized decision-making under coupled constraints. Assigning autonomous agents to individual departmental nodes preserves administrative boundaries, conforms localized actions to domain limitations, and supports inter-agent routing negotiation.

## 4.2 Dec-POMDP Formulation and Theoretical Rationale

We mathematically formulate the hospital-wide coordination problem as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP), defined by the tuple $\mathcal { M } = \langle \mathcal { N } , \mathcal { S } , \mathcal { A } , \mathcal { T } , \mathcal { R } , \mathcal { Q } , \mathcal { O } , \gamma \rangle$ , where $\mathcal { N } = \{ 1 , \dots , M \}$ represents the departmental agent ensemble.

Local Observation Space $\Omega _ { i }$ and Rationale. Due to partial observability, agent $i \in \mathcal N$ does not access the global state $\mathbf { n } \in S$ . Instead, at decision step $t ,$ agent i receives a localized observation vector $o _ { i } ( t ) \in \varOmega _ { i }$ . Rather than streaming high-dimensional telemetry across nodes, local clinical indicators are encoded into a compact 5-dimensional representation:

$$
o _ { i } ( t ) = \left[ o _ { i } ^ { \mathrm { o c c } } ( t ) , o _ { i } ^ { \mathrm { q u e u e } } ( t ) , o _ { i } ^ { \mathrm { a v a i l } } ( t ) , o _ { i } ^ { \mathrm { s l a } } ( t ) , o _ { i } ^ { \mathrm { a l p h a } } ( t ) \right]\tag{3}
$$

which capture the bed occupancy ratio, normalized waiting queue depth, available capacity proportion, Service Level Agreement (SLA) violation warning flags (triggered when patient waiting delay exceeds standard thresholds), and the currently active discharge acceleration intensity, respectively. From a queueing theory perspective, these five indicators perfectly encapsulate the instantaneous trafic utilization parameter $\rho _ { i } ( t )$ and queue derivative $d L _ { q , i } / d t$ , providing sufficient state representation for decentralized action inference without requiring full visibility of distant hospital wards.

Discrete Cooperative Action Space $\mathbf { \mathcal { A } } _ { i }$ and Rationale. Each departmental agent i exercises dual-channel intervention authority over local patient flow through a discrete action space $\mathcal { A } _ { i } = \mathcal { A } _ { i } ^ { \alpha } \times \mathcal { A } _ { i } ^ { \lambda }$ :

– Accelerated Discharge Control (α-Policy, 4 Levels): The agent selects a service acceleration factor $\alpha _ { i } \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 1 5 \}$ to proactively mobilize auxiliary clinical staf, accelerating local treatment completion from baseline $\mu _ { i }$ to $\mu _ { i } ^ { \prime } = ( 1 + \alpha _ { i } ) \mu _ { i }$ . The 15% ceiling conforms to realistic clinical stafing flexibility, while resource expenditures and safety risks are strictly regulated via management and premature-discharge penalties $( W _ { 4 , i }$ and $W _ { 3 , i }$ in Eq. (7)).

Cooperative Routing Integration (λ-Policy, 5 Levels): The agent selects an admission gating factor $\lambda _ { i } \in \{ 0 . 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 \}$ across five discrete levels to scale incoming external and upstream transfer demand $( \lambda _ { i } \cdot \varLambda _ { i } )$ . Here, $\lambda _ { i } ~ = ~ 1 . 0$ maintains unconstrained intake, $\lambda _ { i } ~ = ~ 0 . 0$ executes emergency throttling during acute congestion, and intermediate tiers $( \{ 0 . 2 5 , 0 . 5 0 , 0 . 7 5 \} )$ provide graduated multi-stage regulation (admitting 25%, 50%, or 75%) that avoids binary switching oscillations. Conditioned on internal queue/bed saturation $( o _ { i } ^ { \mathrm { o c c } } , o _ { i } ^ { \mathrm { q u e u e } } )$ and communicated upstream neighbor pressure $( \operatorname* { m a x } _ { j \in \mathcal { N } _ { i } ^ { \mathrm { u p } } } q _ { j } )$ , agents down-modulate $\lambda _ { i }$ to decompress saturated wards and coordinate diversion to step-down units.

Theoretical Design Rationale: Discretizing action dimensions into bounded levels prevents non-stationary policy gradient oscillations during multi-agent negotiation, stabilizing spatial coordination while providing interpretable operational decisions for clinical management.

## 4.3 Topology-Aware Communication and Cooperative Negotiation

In standard independent $\mathrm { M A R L } ,$ agents updating policies locally using only internal observations $o _ { i } ( t )$ experience environmental non-stationarity, because as one unit alters its routing policy, neighboring arrival rates $\lambda _ { i } ( t )$ shift unpredictably. We address this communication discrepancy by implementing a topology-aware coordination mechanism grounded in Mean-field Advantage Actor Critic (MA2C) principles under a Centralized Training with Decentralized Execution (CTDE) architecture [5].

Action Fingerprint Communication Strategy. Instead of establishing unconstrained all-to-all communication networks that scale quadratically with hospital departments, agents share information along directed topological edges derived from our physics prior (Figure 1). During decentralized execution, agent i actively communicates with its adjacent topological neighbors by broadcasting its action fingerprint (FP<sub>i</sub>).

Specifically, each agent globally monitors an Exponential Moving Average (EMA) of its historical policy output distributions, updating via $\mathrm { F P } _ { i } \gets ( 1 -$ $\alpha _ { \mathrm { f p } } ) \mathrm { F P } _ { i } + \alpha _ { \mathrm { f p } } \pi _ { i } ( a _ { i } | o _ { i } )$ , with smoothing factor $\alpha _ { \mathrm { f p } } ~ = ~ 0 . 1$ . At step t, agent i concatenates the incoming action fingerprints of its direct topological neighbors $j \in \mathcal N _ { i }$ into its operational observation space:

$$
\begin{array} { r } { \widetilde { o } _ { i } ( t ) = \big [ o _ { i } ( t ) ; \bigoplus _ { j \in \mathcal { N } _ { i } } \mathrm { F P } _ { j } ( t ) \big ] . } \end{array}\tag{4}
$$

This communicative coupling helps mitigate multi-agent partial observability and stabilizes collaborative training by furnishing adjacent nodes with real-time feedback regarding neighborhood queue trajectories and policy shifts.

Emergence of Cooperative Routing Negotiation. By combining topological action fingerprints with decentralized policy actors, patient routing ceases to be a top-down operational directive and emerges natively from dynamic multiagent negotiation. For instance, when an emergency influx hits the ED agent, its rising queue backlog shifts its action fingerprint $\mathrm { F P } _ { \mathrm { E D } }$ , broadcasting acute upstream pressure to neighboring units. In response, downstream ICU agents sensing the impending arrival surge via communicated fingerprints can cooperatively restrict non-critical elective admissions ("negotiate restriction"), while step-down general medicine wards initiate discharge acceleration $( \alpha _ { j } ~ > ~ 0 )$ to preemptively clear beds for step-down transfers ("negotiate accommodation"). Thus, hospital routing trajectories dynamically stabilize through inter-agent cooperative problem-solving rather than rigid predefined matrices.

## 4.4 Expert-Guided Action Masking and Topological Safety

While cooperative MARL efectively mitigates operational bottlenecks, unconstrained actors maximizing queue rewards could attempt medically unfeasible transfers $( \mathrm { e . g . }$ , routing high-acuity patients to general wards lacking resuscitation infrastructure). To enforce clinical validity, we establish an online expert action masking mechanism derived from MIMIC-IV attending protocols.

During policy execution, candidate routing actions from $\mathcal { A } _ { i } ^ { \lambda }$ are validated against the empirical clinical transition manifold. Any pathway lacking clinical precedence or violating critical care triage guidelines [4] is explicitly pruned before action sampling:

$$
\pi _ { \mathrm { s a f e } } ( a _ { i } | o _ { i } ) = \frac { \pi _ { \theta , i } ( a _ { i } | o _ { i } ) \cdot \mathbb { I } _ { \mathrm { v a l i d } } ( a _ { i } ) } { \sum _ { a ^ { \prime } \in \mathcal { A } _ { i } } \pi _ { \theta , i } ( a ^ { \prime } | o _ { i } ) \cdot \mathbb { I } _ { \mathrm { v a l i d } } ( a ^ { \prime } ) } ,\tag{5}
$$

where $\mathbb { I } _ { \mathrm { v a l i d } } ( a _ { i } ) \in \{ 0 , 1 \}$ indicates clinically permissible destination states. Filtering invalid pathways confines exploratory routing to safe clinical envelopes and reduces policy variance, ensuring adherence to documented care standards.

## 5 Reward Decomposition and Multi-Agent Credit Assignment

In complex multi-agent environments, utilizing a unified global scalar reward across all actors complicates multi-agent credit assignment, as individual agents struggle to discern how localized interventions impact hospital-wide performance. Conversely, relying strictly on localized queue incentives encourages uncoordinated behaviors, such as diverting bottlenecked queues to adjacent wards. We reconcile localized eficiency with system-wide stability by formulating a spatially hierarchical reward decomposition architecture.

At any step $t ,$ the collaborative feedback signal delivered to agent i is separated into three structural layers:

$$
R _ { i } ( t ) = \omega _ { \mathrm { l o c a l } } R _ { \mathrm { l o c a l } , i } ( t ) + \omega _ { \mathrm { n e i g h b o r } } R _ { \mathrm { n e i g h b o r } , i } ( t ) + \omega _ { \mathrm { g l o b a l } } R _ { \mathrm { g l o b a l } } ( t ) .\tag{6}
$$

This hierarchical structure decomposes credit assignment across spatial scales: the local term maintains throughput under resource costs, the neighborhood term suppresses cross-ward spillover, and the global derivative guides macroscopic convergence.

## 5.1 Local Domain Objective $( R _ { \mathrm { l o c a l } , i } )$

The local objective aligns agent behavior with intra-departmental physical efficiency, penalizing immediate queue accumulation while regulating excessive management intervention:

$$
R _ { \mathrm { l o c a l } , i } ( t ) = - \left( W _ { 1 , i } ( t ) + W _ { 2 , i } ( t ) + W _ { 3 , i } ( t ) + W _ { 4 , i } ( t ) - W _ { 6 , i } ( t ) \right) .\tag{7}
$$

Here, the individual operational penalties are formulated as follows: queue backlog penalty $W _ { 1 , i } ( t ) = 1 . 0 \times Q _ { i } ( t )$ , which directly penalizes waiting queue depth $Q _ { i } ( t )$ rather than normal admitted beds; utilization redline penalty $W _ { 2 , i } ( t ) =$ $1 . 0 \times \operatorname* { m a x } ( 0 , u _ { i } ( t ) - 0 . 8 5 ) ^ { 2 }$ , which applies a quadratic penalty when bed utilization $u _ { i } ( t )$ breaches the critical 85% capacity redline; premature-discharge risk $W _ { 3 , i } ( t ) = 1 . 5 \times ( \exp ( 2 . 0 \alpha _ { i } ( t ) ) - 1 )$ , an exponential barrier against unsafe discharge acceleration; management cost $W _ { 4 , i } ( t ) = 0 . 2 \times ( c _ { i } \alpha _ { i } ( t ) )$ , accounting for operational stafing expenses; and local credit bonus $W _ { 6 , i } ( t ) = 0 . 3 \times \mathbb { I } ( \alpha _ { i } ( t ) >$ 0) $\operatorname* { m a x } ( 0 , Q _ { i } ( t - 1 ) - Q _ { i } ( t ) )$ , providing direct positive feedback when active acceleration successfully reduces local waiting lists.

## 5.2 Neighbor Collaborative Objective $( R _ { \mathrm { n e i g h b o r } , i } )$

To discourage localized optimization strategies that simply relieve internal ward congestion by displacing bottlenecked queues onto directly connected departments, we institute a spatial collaborative reward layer regulated by a neighborhood spatial discount factor $( \gamma _ { \mathrm { c o o p } } = 0 . 9 )$

$$
R _ { \mathrm { n e i g h b o r } , i } ( t ) = \frac { \gamma _ { \mathrm { c o o p } } } { | \mathcal { N } _ { i } | } \sum _ { j \in \mathcal { N } _ { i } } R _ { \mathrm { l o c a l } , j } ( t ) .\tag{8}
$$

By incorporating the averaged immediate operational feedback of topologically connected peers, this cooperative objective incentivizes departmental agents to maintain receptive patient transition pathways and coordinate admission timing without requiring global communication bandwidth.

## 5.3 Global Systemic Objective $\left( R _ { \mathbf { g l o b a l } } \right)$

Finally, to anchor all agents to the supreme clinical imperative of hospital-wide flow fluidity, we incorporate a globally shared incremental delay objective based on total network patient delay $D ( t )$

$$
R _ { \mathrm { g l o b a l } } ( t ) = - 0 . 2 \times ( D ( t ) - D ( t - 1 ) ) .\tag{9}
$$

By evaluating the first-order derivative of system delay rather than absolute cumulative delay, this term broadcasts continuous, zero-mean credit assignment signals across the multi-agent ensemble, positively reinforcing cooperative strategies that accomplish macroscopic delay contraction.

## 6 Experiments and Empirical Evaluation

## 6.1 Experimental Configuration and EHR Calibration

Our multi-agent simulation environments are engineered within a high-fidelity Discrete Event Simulation (DES) engine calibrated on the complete MIMIC-IV inpatient cohort of approximately 60 000 patient admissions [11], from which empirical arrival intensities, service distributions, and transition matrices $P$ are parameterized. To evaluate collaborative resilience under operational stress, all experiments simulate stochastic arrival fluctuations under an acute external demand surge multiplier of 1.25.

During the training regimen, the decentralized MA2C agents optimize over 250 simulation episodes, where each episodic epoch represents a continuous operational duration of 4000 hours of simulated hospital workflow. To guarantee broad multi-agent exploration across quantized negotiation tables and thwart premature policy freeze, the policy entropy coeficient is established at 0.05.

## 6.2 Comparative Analysis against Multi-Agent Baselines

To evaluate whether performance improvements stem from genuine multi-agent coordination rather than mere mathematical capacity expansion, we contrast our proposed architecture against four rigorous multi-agent and analytical benchmarks: (1) Static Markovian Baseline $( \alpha = 0 )$ , which models stationary unassisted M/M/c evolution under classical BCMP assumptions; (2) Greedy Heuristic Dispatcher, which grants individual departmental controllers an identical 15% acceleration budget $( \alpha \leq 0 . 1 5 )$ operated via decentralized myopic threshold rules without inter-agent communication; (3) Independent MARL Baseline (IA2C), where independent Advantage Actor-Critic agents optimize individualized policies solely from local observations $o _ { i } ( t )$ without topology-aware action fingerprints or neighbor reward coupling; and (4) Cooperative MA2C (our proposed framework). Both multi-agent approaches (IA2C and MA2C) are evaluated under deterministic (greedy) and exploratory (stochastic) execution modes to assess policy stability under diferent action selection regimes.

As detailed in Table 1, unguided hospital operations under the Static Markovian Baseline yield a severe cumulative delay of 117 478.1 hours over the 4000- hour horizon, driven primarily by acute persistent congestion within the primary Medicine Ward $\left( L _ { q } = 1 2 . 9 5 4 \right)$ . Implementing the Greedy Heuristic Dispatcher moderately reduces cumulative delays to 43 148.5 hours, yet fails to resolve systemic bottlenecks due to myopic reactive interventions (Mean $\alpha = 0 . 0 0 4 )$ that lack spatial coordination with upstream patient trafic surges.

Evaluating the Independent MARL baseline (IA2C) under deterministic versus exploratory execution reveals acute policy sensitivity to action selection regimes. Under stochastic execution, independent actors achieve meaningful delay reduction $( 2 7 8 9 . 7 \pm 6 4 2 . 8 $ hours) and successfully clear queues across General Medicine $( L _ { q } = 0 . 0 0 0 )$ and Med/Surg $( L _ { q } = 0 . 0 0 0 )$ units by actively intervening (Mean $\alpha = 0 . 0 7 5 )$ . However, when operating deterministically under greedy execution without exploratory noise, IA2C experiences catastrophic operational collapse: cumulative delays diverge to $4 6 3 0 3 8 . 7 \pm 2 4 3 6 9 5 . 6$ hours accompanied by severe bottleneck accumulation across all wards $( L _ { q } = 2 . 9 9 6$ in Medicine and $L _ { q } \ = \ 2 . 9 5 4$ in Med/Surg). This instability directly illustrates our theoretical assertions: independent agents learning without neighborhood communicative fingerprints encounter severe environmental non-stationarity, trapped in localized optimization deadlocks and oscillatory routing conflicts when exploratory randomness is removed.

Table 1. System Performance and Departmental Bottleneck Queues across MAS Strategies
<table><tr><td rowspan="2">Policy Strategy</td><td rowspan="2"></td><td rowspan="2">Joint Reward Cumulative Total Delay (hrs) Mean α</td><td rowspan="2"></td><td colspan="3">Mean Bottleneck Queue Depth  $( L _ { q } )$ </td></tr><tr><td>Medicine Ward Med/Surg Emergency (ED)</td><td></td><td></td></tr><tr><td>Cooperative MA2C (Greedy)</td><td> $- 0 . 2 7 8 \pm 0 . 0 0 0$ </td><td> $3 9 7 4 . 0 \pm 4 8 1 . 0$ </td><td>0.044</td><td>0.043</td><td>0.009</td><td>0.636</td></tr><tr><td>Cooperative MA2C (Stochastic)</td><td> $- 0 . 2 9 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 2 0 8 1 . 2 \pm 5 7 1 . 7 }$ </td><td>0.075</td><td>0.000</td><td>0.005</td><td>0.295</td></tr><tr><td>Independent A2C (Greedy)</td><td> $- 0 . 2 7 4 \pm 0 . 0 7 2$ </td><td>463038.7 ± 243695.6</td><td>0.061</td><td>2.996</td><td>2.954</td><td>1.684</td></tr><tr><td>Independent A2C (Stochastic)</td><td> $- 0 . 6 0 8 \pm 0 . 0 0 1$ </td><td>2789.7 ± 642.8</td><td>0.075</td><td>0.000</td><td>0.000</td><td>0.630</td></tr><tr><td>Greedy Heuristic (α ≤ 0.15)</td><td></td><td>43148.5</td><td>0.004</td><td>2.242</td><td>1.869</td><td>0.850</td></tr><tr><td>Static Markovian Baseline (α = 0)</td><td></td><td>117478.1</td><td></td><td>12.954</td><td>1.869</td><td>0.850</td></tr></table>

In contrast, the Cooperative MA2C architecture demonstrates superior and robust congestion mitigation across both evaluation modes, avoiding the instability of uncoordinated learning. Across exploratory stochastic evaluations, the proposed multi-agent framework confines cumulative hospital delay to 2081.2 ± 571.7 hours (and $3 9 7 4 . 0 \pm 4 8 1 . 0$ hours under deterministic greedy execution), yielding an approximate 56-fold reduction compared to static Markovian approximations and outperforming independent MARL actors across both operational regimes. This performance is achieved with consistent overall discharge acceleration (Mean $\alpha = 0 . 0 7 5 )$ . By leveraging inter-agent communicative coordination and topological physical priors, cooperative policies actively divert secondary transfers away from saturated nodes, eliminating queue accumulation across the General Medicine ward $\left( L _ { q } = \mathbf { 0 . 0 0 0 } \right)$ and nearly eliminating it in Med/Surg units $\left( L _ { q } = \mathbf { 0 . 0 0 5 } \right)$ , while registering the lowest bottleneck queue depth within the Emergency Department $\left( L _ { q } = \mathbf { 0 . 2 9 5 } \right)$

## 6.3 Structural Ablation Studies

We quantify the contribution of each architectural component within our proposed framework through structural ablation experiments by isolating core mechanisms from the complete Cooperative MA2C model. To isolate architectural component eficacy from action-selection variance across execution regimes, all ablation variants and the reference model are evaluated across an independent set of standardized random evaluation seeds (accounting for the empirical variance between Table 1 and Table 2).

Table 2. Structural Ablation Evaluation of Multi-Agent Framework Components
<table><tr><td>Ablated Configuration Architecture</td><td colspan="3">Cumulative Delay (hrs) Mean Speedup (α) ICU Premature Discharge Rate (%)</td></tr><tr><td>Full Cooperative MA2C (Proposed)</td><td> $\mathbf { 2 8 3 5 . 2 \pm 9 7 2 . 1 }$ </td><td>0.044</td><td>0.211</td></tr><tr><td>(a) w/o Neighbor Action Fingerprints</td><td> $3 0 2 4 . 5 \pm 8 9 3 . 0$ </td><td>0.037</td><td>0.227</td></tr><tr><td>(b) w/o Cooperative Routing (λ-disabled)</td><td> $2 2 6 2 3 1 . 1 \pm 2 3 8 6 1 . 5$ </td><td>0.051</td><td>0.368</td></tr><tr><td>(c) w/o Dynamic Acceleration (α-disabled)</td><td> $3 9 9 0 . 8 \pm 8 7 7 . 6$ </td><td>0.000</td><td>0.212</td></tr><tr><td>(d) w/o Expert Prior Action Masking</td><td> $3 1 0 8 . 9 \pm 6 7 6 . 8$ </td><td>0.044</td><td>0.190</td></tr></table>

The ablation findings presented in Table 2 substantiate our architectural design choices: removing neighbor action fingerprints (a) elevates cumulative delay from 2 835.2 to 3 024.5 hours while slightly dampening discharge acceleration (Mean $\alpha = 0 . 0 3 7 )$ , indicating that active topology-aware signaling assists agents in anticipating incoming trafic shifts and mitigating Dec-POMDP nonstationarity; disabling cooperative routing (b) drives overall delay into acute divergence (226 231.1 hours) alongside an elevated premature discharge rate (0.368%), demonstrating that localized capacity scaling alone cannot bypass structural network bottlenecks without spatial routing negotiation; disabling dynamic acceleration (c) increases cumulative delay to 3 990.8 hours (Mean α = 0.000), confirming that auxiliary capacity elasticity is essential to relieve peak localized surges and maintain stable throughput; finally, removing expert prior action masking (d) increases system delay to 3 108.9 hours with an observed premature discharge rate of 0.190%. Without explicit empirical priors bounding the transition manifold, unguided actors adopt misaligned routing strategies, such as overly restrictive transfer policies that artificially compress local ICU throughput while displacing congestion upstream, confirming the necessity of expert action masking to ensure balanced operational eficiency and adherence to documented clinical standards.

## 6.4 Clinical Pathway Safety and Anomaly Verification

A resilient medical AI system must enhance operational scheduling without introducing anomalous clinical pathways. To verify clinical safety, we monitored all patient trajectories throughout simulation execution, quantifying two major pathway violations against historical empirical baselines in MIMIC-IV: pingpong transfers, representing cyclical, redundant ward transitions (e.g., Ward A → Ward B → Ward A) that sever care continuity; and premature ICU discharges, capturing high-risk events where unstable critical care patients are discharged directly out of Intensive Care Units without traversing required step-down wards.

Table 3. Empirical Validation of Clinical Pathway Safety against Real Hospital Records
<table><tr><td>Operational Policy Benchmark</td><td colspan="2">Ping-Pong Transfer Rate (%) Premature ICU Discharge Rate (%)</td></tr><tr><td>Real Historical Data (MIMIC-IV Records</td><td>7.375</td><td>2.585</td></tr><tr><td>Independent MARL Baseline (IA2C, Greedy Execution)</td><td>2.769</td><td>0.298</td></tr><tr><td>Independent MARL Baseline (IA2C, Stochastic Execution)</td><td>0.674</td><td>0.095</td></tr><tr><td>Cooperative MA2C (Greedy Execution)</td><td>1.727</td><td>0.214</td></tr><tr><td>Cooperative MA2C (Stochastic Execution)</td><td>1.736</td><td>0.211</td></tr></table>

As documented in Table 3, operational trajectories in MIMIC-IV historical records exhibit a baseline ping-pong transition rate of 7.375% and a premature ICU discharge incidence of 2.585%. Here, a ping-pong transfer is identified when a patient revisits the same ward $\geq ~ 2$ times within a single admission $( A \to B \to A )$ , while premature ICU discharge denotes transfers occurring before completing the department’s calibrated Average Length of Stay $\left( \mathrm { A L O S } _ { \mathrm { I C U } } \right)$ All learned policies substantially reduce both anomaly types relative to historical baselines. Independent MARL actors (IA2C) achieve the lowest ping-pong transfer rates (0.674% under stochastic execution) due to conservative routing. Our Cooperative MA2C framework records moderately higher ping-pong rates (1.727% and 1.736% under greedy and stochastic execution, respectively), reflecting its active cooperative routing strategy that redistributes patients across wards to relieve congestion bottlenecks. Importantly, both MA2C execution modes maintain premature ICU discharge rates (0.214% and 0.211%) comparable to IA2C stochastic (0.095%) and substantially below historical baselines (2.585%), confirming that the expert action mask efectively preserves clinical safety even under dynamic cooperative routing.

## 7 Discussion & Conclusion

A common abstraction in healthcare operations modeling is the implicit assumption of a linear relationship between incoming patient trafic demand and emergent hospital congestion, namely, that a 10% increase in patient arrivals will induce an approximately proportional 10% expansion in waiting times.

However, classical queueing theory reveals a non-linear relationship. Canonical queuing formulations, such as the Erlang-C model for $\mathrm { M } / \mathrm { M } / \mathrm { c }$ multi-server queues and the Pollaczek–Khinchin formula for $\mathrm { M } / \mathrm { G } / 1$ systems, indicate that expected waiting queue depths grow according to an asymptotic inverse relationship with departmental resource utilization:

$$
{ L _ { q } } \propto \frac { 1 } { 1 - \rho _ { i } }\tag{10}
$$

As illustrated in Figure 2, when a clinical unit operates within moderate utilization boundaries $( \rho _ { i } \ \leq \ 0 . 7 )$ , the department resides on a stable operating plateau where stochastic arrival increases yield negligible wait-time escalation. However, once operational utilization approaches the physical capacity limit $( \rho _ { i } \to 1 ^ { - } )$ , the denominator $( 1 - \rho _ { i } )$ approaches zero, driving queue accumulation and waiting delay into a superlinear asymptotic surge. Within this congestive regime, minor arrival perturbations can trigger inter-departmental blocking cascades.

This relationship explains why stationary approximations degrade under heavy load and illustrates the role of multi-agent coordination in dynamic healthcare optimization. Under static modeling assumptions, high-demand wards such as General Medicine operate continuously near saturation $\left( \rho _ { i } \approx 0 . 9 9 9 \right)$ , generating an accumulated simulation delay of 117 478.1 hours. Because individual wards experience localized queue surges independently, neither centralized scheduling policies nor uncoordinated routing rules react efectively to these non-linear transitions without inducing inter-departmental bottlenecks.

![](images/4ef100b4202b36ed87823b095818fc6221dcdbb364af51e629f4f07ba7ac34a7.jpg)  
Fig. 2. Asymptotic queue escalation as department utilization $\rho _ { i } \to 1 ^ { - }$ , illustrating how localized multi-agent interventions actively transition operations back into the stable operating plateau.

Our approach mitigates these bottlenecks through cooperative multi-agent decision-making. By coupling localized departmental actors with topological action-fingerprint communication, the policy network maintains responsiveness to regional congestion gradients. Instead of requiring global stafing additions, agents negotiate localized interventions by combining slight service acceleration (Mean $\alpha = 0 . 0 7 5 )$ with cooperative patient routing diversion. This multi-agent policy action addresses the mathematics of the asymptotic curve: introducing temporary service speedup to a heavily utilized ward reduces its efective utilization below the non-linear inflection threshold $( \rho _ { i } \to 1 ^ { - } )$ , restoring operations to the linear delay regime. As a result, timed cooperative interventions accomplish a 56-fold reduction in simulation delay (2 081.2 hours), validating the practical utility of decentralized multi-agent coordination.

Limitations and Future Research Horizons. Although our proposed framework demonstrates efective physics-informed multi-agent coordination, open challenges remain. Currently, baseline service capacities $\mu _ { i , t }$ rely on historical averages from MIMIC-IV and aggregated clinical stage identifiers, overlooking how real-time clinician workload, nursing fatigue, and workplace stress dynamically alter service eficiency. To bridge this simulation-to-reality gap, future research will explore a hierarchical hybrid framework coupling macroscopic MARL routing with microscopic behavioral simulation. Specifically, embedding LLM-based cognitive agents (LLM-Agents) can simulate real-time clinician reasoning and staf behavioral dynamics under stress, ofering a promising path toward interpretable decision support and resilient operational governance across complex healthcare institutions.

## References

1. Armony, M., Israelit, S., Mandelbaum, A., Marmor, Y.N., Tseytlin, Y., Yom-Tov, G.B.: On patient flow in hospitals: A data-based queueing-science perspective. Stochastic systems 5(1), 146–194 (2015)

2. Barz, C., Rajaram, K.: Elective patient admission and scheduling under multiple resource constraints. Production and Operations Management 24(12), 1907–1930 (2015)

3. Bekker, R., Koeleman, P.M.: Scheduling admissions and reducing variability in bed demand. Health care management science 14(3), 237–249 (2011)

4. Blanch, L., Abillama, F.F., Amin, P., Christian, M., Joynt, G.M., Myburgh, J., Nates, J.L., Pelosi, P., Sprung, C., Topeli, A., et al.: Triage decisions for icu admission: report from the task force of the world federation of societies of intensive and critical care medicine. Journal of critical care 36, 301–305 (2016)

5. Chu, T., Wang, J., Codecà, L., Li, Z.: Multi-agent deep reinforcement learning for large-scale trafic signal control. IEEE transactions on intelligent transportation systems 21(3), 1086–1095 (2019)

6. Ekpo, P.O., La, B., Wiener, T., Agarwal, S., Agrawal, A., Gonzalez-Pumariega, G., Molu, L.P., Taylor, A.: Skill-aligned fairness in multi-agent learning for collaboration in healthcare. arXiv preprint arXiv:2508.18708 (2025)

7. Funhiro, W.: Standardization And Strengthening The Functionality Of Hospital Management Boards In Central Hospitals Of Zimbabwe. Ph.D. thesis, University of KwaZulu-Natal, Westville (2020)

8. Guido, R., Groccia, M.C., Conforti, D.: An eficient matheuristic for ofline patientto-bed assignment problems. European Journal of Operational Research 268(2), 486–503 (2018)

9. Hao, Q., Xu, F., Chen, L., Hui, P., Li, Y.: Hierarchical multi-agent model for reinforced medical resource allocation with imperfect information. ACM Transactions on Intelligent Systems and Technology 14(1), 1–27 (2022)

10. Jebbor, S., El Afia, A., Chiheb, R.: An approach by human and material resources combination to reduce hospitals crowding. International Journal of Pervasive Computing and Communications 15(2), 58–79 (2019)

11. Johnson, A.E., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T.J., Hao, S., Moody, B., Gow, B., et al.: Mimic-iv, a freely accessible electronic health record dataset. Scientific data 10(1), 1 (2023)

12. Laskowski, M., McLeod, R.D., Friesen, M.R., Podaima, B.W., Alfa, A.S.: Models of emergency departments for reducing patient waiting times. PloS one 4(7), e6127 (2009)

13. Lei, X., Na, L., Xin, Y., Fan, M.: A mixed integer programming model for bed planning considering stochastic length of stay. In: 2014 IEEE International Conference on Automation Science and Engineering (CASE). pp. 1069–1074. IEEE (2014)

14. Long, E.F., Mathews, K.S.: The boarding patient: efects of icu and hospital occupancy surges on patient flow. Production and operations management 27(12), 2122–2143 (2018)

15. Lv, J., Kim, B.G., Li, K., Lu, H.: Multi-agent reinforcement learning driven dynamic resource optimisation in healthcare transportation networks. CAAI Transactions on Intelligence Technology 11(2), 316–331 (2026)

16. Mizuno, S., Ohba, H.: Resource allocation optimization for disaster medical systems based on open bcmp queueing networks. Operations Research, Data Analytics and Logistics p. 200505 (2026)

17. Schmidt, R., Geisler, S., Spreckelsen, C.: Decision support for hospital bed management using adaptable individual length of stay estimations and shared resources. BMC medical informatics and decision making 13(1), 3 (2013)

18. Seid, A.M., Erbad, A., Abishu, H.N., Albaseer, A., Abdallah, M., Guizani, M.: Multiagent federated reinforcement learning for resource allocation in uav-enabled internet of medical things networks. IEEE Internet of Things Journal 10(22), 19695– 19711 (2023)

19. Sheikh, A., Chong, E.K.: Advancing aiomt-enabled healthcare system-of-systems using multi-agent reinforcement learning. IEEE Access (2025)

20. Thomas, B.G., Bollapragada, S., Akbay, K., Toledano, D., Katlic, P., Dulgeroglu, O., Yang, D.: Automated bed assignments in a complex and dynamic hospital environment. Interfaces 43(5), 435–448 (2013)

21. Vainieri, M., Panero, C., Coletta, L.: Waiting times in emergency departments: a resource allocation or an eficiency issue? BMC health services research 20(1), 549 (2020)

22. Vancroonenburg, W., De Causmaecker, P., Vanden Berghe, G.: A study of decision support models for online patient-to-room assignment planning. Annals of Operations Research 239(1), 253–271 (2016)