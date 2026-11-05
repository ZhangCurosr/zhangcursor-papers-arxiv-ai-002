# RAGAM-AI: A Hybrid Reinforcement Learning based Agentic

approach powered by Generative AI for Edge Resource Management

Shreya K. Chari , Student Member, IEEE, John S. Vardakas , Senior Member, IEEE, Kostas Ramantas , Christos Verikoukis , Senior Member, IEEE

Abstract—Resource management for distributed edge networks hosting diferent services has been handled eficiently using Reinforcement learning (RL). Reward generalization in complex workflows is still an open issue within the RL realm. Therefore, this work proposes RAGAM-AI, a hybrid framework that combines the capabilities of RL algorithms and Generative-AI (Gen-AI) for powering an Agentic AI workflow that optimizes resource allocation at the Multi-access Edge Computing (MEC) units. This framework further resolves the reward generalization issue by utilizing Large Language Model (LLM) for feedback. A Multi-Agent Soft-Actor Critic (MASAC) using MathΣtral as a reward oracle is implemented for managing resource allocations at the MEC. RAGAM-AI interprets system state information and evaluates multiobjective trade-ofs guiding the resource orchestration loop over distributed edge server agents. The complexity of the framework is further brought down by incorporating surrogate reward model. The overall performance is evaluated against the single agent SAC, REINFORCE and random policies for benchmarking. On using RAGAM-AI, lower allocation deviations for CPU, memory and bandwidth resources are reported while minimizing the latency and data rate costs. A scalable AI as a feedback (AIAF) solution optimizing network resources with significant performance improvement is thus presented.

Index Terms—Reinforcement learning (RL), Generative-AI (GenAI), Agentic AI, Resource management, Multi-access Edge Computing (MEC), Reinforcement Learning from AI Feedback (RLAIF)

## I. Introduction

increasing latency and congestion within networks. In such a scenario, a MEC at the edge of the network emerged as a solution for bringing computation and devices closer enabling task ofloading in resource-constrained conditions [1]. MEC is further in control of network resources such as compute power, bandwidth and memory. However, eficient resource management within MEC yet remains as a critical and challenging problem.

The long-established systems fail to consider nonstationary network trafic distributions and the coupling efect between diferent resources [2]. Heterogeneous nodes now cater to diverse use-cases that require low-latencies and faster compute capacities. Many of these issues are resolved at the edge network with the help of MEC servers. Long transmission time, energy expenditures and data breaches are alleviated by partially or fully ofloaded tasks to the edge [3]. On the other hand, edge caching relieves delays caused from the backhaul pressure significantly by fetching frequently accessed data [4]. Fault tolerance and system resiliency can be improved by having task distribution and decision-making closer to the network rather than a centralized cloud [5]. There is an enhanced context-awareness as information with regards to local radio link signals and network trafic load bursts become more available [6].

There are numerous use-cases that benefit from putting edge computing into operation. In Vehicle-to-everything (V2X), smart transportation systems’ edge computing has enabled low latency and high reliability safety services [7]. Edge networks can support real-time fault-detection and load balancing in smart grids alleviating network outage issues [8]. For running inferences, analytics and predictive maintenance, Industrial IoT (IIoT) often rely on edge powered solutions [9]. Edge caching is often used for Extended Reality (XR) technologies for providing rendering support [10]. Digital twins require localized optimization for data-ingestion and for processing sensor outputs relying on edge support [11].

However, a wide range of open challenges continues to limit practical deployment of these use-cases. Various network aspects such as compute, storage, communication and energy are optimized in isolation due to independently functioning Radio Access Network (RAN) and MEC orchestrators [16]. Network handover and service migration are not yet completely service-aware, causing disruption in providing quality services to the end user [13]. The privacy and security concerns at the edge in distributed settings are also issues of concern in order to make the systems more fault resilient [14]. Limitations on the standards and the available 5G architectures make the cross-domain interactions complex leading to unsatisfied Service Level Agreements (SLA) [15].

RL has emerged as a promising solution to bridge some of the gaps mentioned above. Resource allocation is now possible subject to dynamic network conditions using RL’s learning abilities even under fluctuating workloads [16]. Multi-objective optimization involving both the communication and compute requirements can be achieved using RL [17]. Decentralized distributed multi-agent RL system enables localized learning at each node contributing to global objectives in dense edge deployments [18]. RLbased approaches have also showcased reduction in overall latencies and energy expenditures with better resource utilization [19].

The advantages of using RL algorithms at the edge got further enhanced with the availability of Generative modeling. RL algorithms often are constrained by partial observability, limited training data and lack of explainability. Generative models can assist RL algorithms by synthesizing unforeseen scenarios and help in generalization [20]. RL algorithms involve interactive learning with the environment often making it sample ineficient and expensive [21]. Gen-AI aided intervention can alleviate this problem by optimizing the search spaces and reducing some of the interactions.

In recent times, there’s an increasing interest in autonomous frameworks that can manage and execute sophisticated decisions entirely on its own. These systems give rise to concepts like Agentic-AI systems and Mixtureof-Experts (MoE) which breakdown complex tasks into independent sub-tasks at both the training and evaluation level [22]. Further, the performance enhancement is achieved using valuable feedback through methods like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF) [23, 24]. In RLHF, direct human assessment is included as a feedback creating scalability and expense issues. On the other hand, RLAIF involves AI assistants for supervising the algorithm behaviour.

Keeping in mind all the discussion above, it can be established that resource management at the network edge is still a fundamental challenge as the networks becomes more heterogeneous with diverse performance constraints. RL algorithms provide a promising relief for adaptive decision-making while still coping with reward bias and explainability challenges. The studies [25, 26] confirm that imperfect or rigid reward formulations can indeed afect the learning capabilities of the RL policies. These issues are especially relevant in edge resource orchestration, where static reward shaping may inadequately capture dynamic trade-ofs among latency, utilization, fairness, and demand variability. Recent advances in Gen-AI when combined with existing RL solutions can provide the required performance enhancement improving the sample eficiency and the semantic reasoning for generalization.

Therefore, in this work, a hybrid framework utilizing the learning capabilities of a distributed Multi-Agent Deep RL (MADRL) at the MEC is integrated with the feedback from GenAI-based LLM model extending the preliminary results in [27]. The framework incorporates multiple interacting RL agents operating across edge nodes driven by feedback from the LLM. The policy updates are coordinated through shared environmental objectives and reward signals. The framework therefore showcases core agentic properties of autonomy, feedback inclusion, task distribution and adaptive behavior. The framework thus achieved is interpreted as an agentic-AI system successful in eliminating the reward bias and in improving the overall algorithm performance measured in terms of eficient allocations, latency and data rate. The salient features of the framework implemented are as follows:

TABLE I: Summary of results
<table><tr><td rowspan=1 colspan=1>PolicyMetrics↓</td><td rowspan=1 colspan=1>RAGAM-AI</td><td rowspan=1 colspan=1>SAC</td><td rowspan=1 colspan=1>REINFORCE</td><td rowspan=1 colspan=1>Random</td></tr><tr><td rowspan=1 colspan=1>CPU dev.</td><td rowspan=1 colspan=1>23.65%</td><td rowspan=1 colspan=1>28.61%</td><td rowspan=1 colspan=1>32.53%</td><td rowspan=1 colspan=1>44.93%</td></tr><tr><td rowspan=1 colspan=1>Memory dev.</td><td rowspan=1 colspan=1>20.36%</td><td rowspan=1 colspan=1>24.35%</td><td rowspan=1 colspan=1>24.68%</td><td rowspan=1 colspan=1>46.37%</td></tr><tr><td rowspan=1 colspan=1>Bandwidth dev.</td><td rowspan=1 colspan=1>20.40%</td><td rowspan=1 colspan=1>25.27%</td><td rowspan=1 colspan=1>26.5%</td><td rowspan=1 colspan=1>46.92%</td></tr><tr><td rowspan=1 colspan=1>Latency cost</td><td rowspan=1 colspan=1>6.70e-3</td><td rowspan=1 colspan=1>6.72e-3</td><td rowspan=1 colspan=1>6.724e-3</td><td rowspan=1 colspan=1>8.12e-3</td></tr><tr><td rowspan=1 colspan=1>Data rate cost</td><td rowspan=1 colspan=1>1.151e+8</td><td rowspan=1 colspan=1>1.153e+8</td><td rowspan=1 colspan=1>1.154e+8</td><td rowspan=1 colspan=1>1.228e+8</td></tr></table>

1) A RL-based SAC multi-agent system using AI as feedback by utilizing the MathΣtral driven reward oracle is wrapped in a hybrid Agentic AI framework.

2) The developed RAGAM-AI algorithm focuses on managing edge network resources: CPU, memory and bandwidth eficiently.

3) Resource deviation, latency cost and data rate cost is used for performance evaluation. The results show that RAGAM-AI outperforms the considered benchmarking policies. The average values for improvements observed are summarized in Table I.

4) The problem of reward bias is alleviated by the inclusion of LLMs in the reward oracle which is further optimized by training a supervised reward model.

The remainder of this paper is organized as follows. Section II discusses the related works focusing on diferent approaches for resource management at network edge. Section III gives an overview of the system design used for the proposed framework in this paper. Section IV lays out the details of the framework and algorithm thus designed with relevant background required for understanding. The performance results and comparisons are further analyzed in Section V. Towards the end, Section VI provides conclusion and future research directions with respect to this paper.

## II. Related work

In this section, the related literature prior to this work is reviewed and summarized. This section attempts to showcase the evolution of resource management at the network edge from the initial traditional optimization approaches to the current amalgamation with AI-guided management. Further, the strengths, highlights and limitations of these works are discussed with a summary in Table II and III. The section ends with a careful consideration of how the research idea in this work fills the gaps in the literature by proposing significant improvements.

## A. Edge resource management

Attempts have been made using traditional optimization techniques to manage and orchestrate the edge network resources in various works. The authors of [28], propose a multi-objective evolutionary optimization framework to resolve the task ofloading and resource allocation issues at the MEC servers. The developed decompositionbased evolutionary algorithm thus developed minimizes average task completion time and energy consumption of mobile devices. A MEC architecture catering to Unmanned Aerial Vehicles (UAVs) and Device-to-Device (D2D) communication for eficient resource utilization is proposed in [29]. A joint optimization problem integrating trajectory design, resource allocation, and task ofloading decisions is formulated minimizing computation latency.

TABLE II: Summary of related work
<table><tr><td colspan="2">Reference</td><td>Application</td><td colspan="4">Highlights</td><td rowspan="2">Metric evaluated</td></tr><tr><td>[16]</td><td></td><td>Network slicing</td><td>Resource allocation</td><td>Task offloading</td><td>Expandability ①</td><td>Heterogeneity C</td></tr><tr><td>[28</td><td></td><td>IoT devices</td><td></td><td></td><td>O</td><td>O</td><td>Processing ratio Task completion time, energy consumption</td></tr><tr><td></td><td>[29</td><td>UAV, D2D</td><td></td><td></td><td></td><td>1</td><td>Task offloading ratio, computational latency</td></tr><tr><td></td><td>30</td><td>Distributed systems</td><td></td><td></td><td></td><td></td><td>System gain, delay, energy</td></tr><tr><td></td><td></td><td>Ultra-dense low latency networks</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>[31]</td><td></td><td></td><td></td><td></td><td></td><td>Resource utilization, load reduction, energy</td></tr><tr><td></td><td>[32]</td><td>Latency sensitive users Cloud-native orchestration</td><td></td><td></td><td></td><td>①</td><td>Resource utilization, task completion time &amp; offloading proportion</td></tr><tr><td></td><td>[33] [34]</td><td></td><td></td><td></td><td></td><td></td><td>Resource allocation, latency</td></tr><tr><td></td><td>[35]</td><td>Mobile user offloading</td><td></td><td></td><td></td><td></td><td>Workload balance, system cost</td></tr><tr><td></td><td>36]</td><td>Digital twin</td><td></td><td></td><td></td><td>O</td><td>System cost, latency cost, energy cost</td></tr><tr><td></td><td>37</td><td>IoT</td><td></td><td></td><td></td><td>O</td><td>Execution utility &amp; time</td></tr><tr><td></td><td>[38</td><td>Vehicular edge computing</td><td></td><td></td><td></td><td>一</td><td>Inference time, backlogged computations</td></tr><tr><td></td><td>[39</td><td>Blockchain-mining Augmented reality</td><td></td><td></td><td></td><td>一</td><td>Offloading utility, Network bandwidth cost, system utility</td></tr><tr><td></td><td>[40]</td><td>SAGINs</td><td></td><td></td><td></td><td></td><td>Utility, frame rate, accuracy, latency</td></tr><tr><td></td><td>[41]</td><td>Cloud-edge collaboration</td><td></td><td></td><td></td><td></td><td>Energy, latency</td></tr><tr><td></td><td>[42</td><td>AI-generated content</td><td></td><td></td><td></td><td>1</td><td>Task performance, QoE</td></tr><tr><td></td><td>[43]</td><td>Consumer electronic devices</td><td></td><td></td><td></td><td>O</td><td>Hit ratio, objective function performance</td></tr><tr><td></td><td>[44]</td><td>Metaverse</td><td></td><td></td><td></td><td></td><td>Cost of renting, makespan, energy consumption Reward, task performance</td></tr></table>

: Fully supported, : Partially supported, : Not supported

TABLE III: Limitations & improvement from related work
<table><tr><td>Reference</td><td colspan="2">Identified limitations</td><td colspan="2">Improvement using RAGAM-AI</td></tr><tr><td>[28]- [32]</td><td></td><td>• Scalability concerns • Static or semi-static network assump- tions • Security and privacy Concerns • High computational complexity • Parameter sensitivity</td><td>• Model-free optimization spaces • Automatic policy learning</td><td>• Improved handling of high-dimensional</td></tr><tr><td>[16], [33]- [40]</td><td>layers</td><td>• Information exchange during centralized training • Training instabilities from hierarchical • Hybrid-action spaces • Synchronization issues • High algorithmic complexity • Deployment and Integration Challenges</td><td colspan="2">• Sequential decision-making • Adaptation to changing environment • Elimination of labelled datasets • Focus on long-term objectives • Distributed learning</td></tr><tr><td>[41]- [44]</td><td></td><td>• Model maintenance and updates • Computational overhead at the edge • Stationarity considerations</td><td>Improved reasoning • Avoiding reward bias • Including AI as feedback</td><td>Handling complex state information</td></tr></table>

In [30], user Quality of Experience (QoE) in federated MEC systems is jointly optimized using service caching, computational resource allocation, and task ofloading. A novel two-stage algorithmic framework based on approximation and decomposition techniques framed as a Mixed Integer Non Linear Programming (MINLP) with multilayer constraints. A resource-eficient joint service caching and workload scheduling framework for ultra-dense edge networks is proposed in [31]. The developed CoShare algorithm reduces the workload on the cloud by utilizing Lyapunov technique for optimization. The research work in [32] uses genetic algorithm for joint optimization of ofloading ratios, wireless bandwidth allocation, and computational resource distribution. Simulation results demonstrates reduction in task completion time while maintaining performance.

## B. AI-driven resource optimization using RL

Over time, many of the open challenges were mitigated by making use of RL advancements. A two-level slicing model is introduced in [16] for dynamic networkslicing and resource allocation. The network slicing is resolved with a QMIX-based RL algorithm while the resource allocation is resolved using a Proximal Policy Optimization (PPO) based algorithm. In [33], a cloudnative architecture explores resource allocation eficiency and network resource utilization. The architecture is validated by testing it on a testbed to demonstrate the improvements. The authors of [34] use hierarchical RL to tackle joint caching and resource allocation issues in MEC systems. In recent times digital twins have gained popularity leading to high compute requirements at the edge. In [35], a federated deep RL scheme is designed to optimize task ofloading, configurations of passive reflecting links, and computing resources from a twolayer digital twin edge network. A Knowledge-Defined Edge Computing Networks (KDECN) architecture along with a RL-based Long-term Computation Ofloading and computation Resource Allocation (L-CORA) algorithm is presented in [36]. This approach intends to manage largescale networks ensuring long-term quality of computing service for user devices.

In the context of Vehicular Edge Computing (VEC), Multi-Agent Difusion-based Deep RL (MAD2RL) is proposed in [37] to jointly optimize Deep Neural Net (DNN) partitioning, task ofloading, and resource allocation. In addition to that, a Lyapunov optimization technique is initially used to decouple the long-term problem into a per-slot deterministic problem. A novel solution combining blockchain-mining and edge system improving the overall system utility is presented in [38]. The multi-objective formulation thus developed jointly optimizes ofloading decision, channel selection, transmit power allocation, and computational resource allocation. In [39], low-latency video communication ensuring video quality in Mobile Augmented Reality (MAR) applications is explored. A Gated Recurrent Unit-Augmented Twin Delayed Deep Deterministic Policy Gradient (GTD3) is used to resolve the challenges from the need for dynamic computational resources and limited availability of network bandwidth. The authors of [40] propose a novel MADRL algorithm in the MEC infrastructure within Space-Air-Ground Integrated Networks (SAGINs). The ground user tasks are modelled as Directed Acyclic Graphs (DAGs) while reducing energy consumption and latency in MEC.

## C. GenAI augmented performance enhancement

Introduction of GenAI applications in RL-based systems has brought tremendous shift in how intelligent systems learn, adapt, and make decisions. For instance, in [41], an innovative framework called EdgeNetLLM is developed for eficiently adapting LLM models for the tasks of adaptive bitrate streaming (ABR) and cluster job scheduling (CJS) in mobile networks. Additionally, it can also adapt to specific tasks and inputs minimizing the overall costs and latency. In [42], a Two-Timescale Deep RL (T2DRL) algorithm is decomposed as a caching subproblem on a long-timescale and a resource allocation subproblem on a short-timescale. The algorithm is designed for resource provisioning in edge-enabled AIGC service for reducing latency. Similarly, the authors of [43] propose a lightweight LLM to augment RL for independent task scheduling minimizing makespan, operational costs, and energy consumption. To address the issue of GenAI models being resource-intense, [44] present a AI-Generated Optimal Decision (AGOD) algorithm for optimally select service providers. The AGOD algorithm utilizes Deep Difusion Soft Actor-Critic (D2SAC) for increasing the eficiency of the selection made.

![](images/14469c89806ac0488c875019328d11ae4bbea29f834544f5da806d76611d0291.jpg)  
Fig. 1: System design

## D. Research gap & contributions

This work improves the state-of-the-art literature by emphasizing on the need for performance enhancement of existing RL algorithms by taking advantage of the advancements and availability of pre-trained LLMs. RAGAM-AI is a model-free multi-agent approach that can function under dynamic network connections. The RL policy within RAGAM-AI can be conveniently scaled to multiple servers tackling training instabilities and growing state space. Further, the centralized training and decentralized execution within the algorithm enables distributed learning. RL algorithms are often entirely driven by the reward signals, thus a poorly designed signal either introduces learning bias or amplifies the ground truth. Therefore, RAGAM-AI uses a reward oracle that combines a traditional mathematically backed reward function alongside reward signals received from mindful prompting to the MathΣtral model to minimize the reward generalization issue. Further, a small Deep Neural Network (DNN) is trained to mimic the reward function for lightweight architecture. The crucial contributions are listed below:

1) Decentralized training mechanism by making use of MASAC algorithm with of-policy learning enabling scalability and sample eficiency.

2) Improved decision-making via hybrid Agentic AI formulation using the capabilities of interactive strategy updates through RL policies and contextbased reasoning feedbacks from LLMs.

3) Optimized resource allocations minimizing the deviations from the resource demand while providing enhanced latency and data rate.

## III. System Design

In this section, the system model outline is provided, followed by the formal definition of the RAGAM-AI algorithm. The symbols used throughout this work are summarized in Table IV. Additionally, Fig.1 paints an overall visual how the system design looks with the most important components and their functionalities. The system focuses mainly on the resource requests received at the edge servers and the RL agents placed at each of the considered servers. The user requests are translated as edge-level resource utilization metrics, specifically CPU, memory, and bandwidth consumption through network services. Admission control is beyond the scope of this work, and it is assumed that all requests are admitted for convenience.

## A. Problem definition

The resource request intent from the users is mapped to the MEC servers $n _ { i }$ as compute c, memory m and bandwidth b demands. At each time instance t the request received can be visualized as in (1). The requests are associated with network slices which are then admitted and orchestrated by at the edge.

$$
\begin{array} { r } { \left[ \begin{array} { c c c c } { ( c , m , b ) _ { t } ^ { n _ { 1 } } } & { ( c , m , b ) _ { t } ^ { n _ { 2 } } } & { \cdots } & { ( c , m , b ) _ { t } ^ { N } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { ( c , m , b ) _ { T } ^ { n _ { 1 } } } & { \cdots } & { \cdots } & { ( c , m , b ) _ { T } ^ { N } } \end{array} \right] } \end{array}\tag{1}
$$

The agents at each MEC server are constantly calculating the latency L and data rate D using Shannon equation in (2) and (3) to evaluate the agents decision making abilities [45]. The SNR is calculated using channel gains and the noise power $N P = - 1 7 4 + 1 0 \log _ { 1 0 } B _ { f } - N F$ where $N F$ is the noise figure. Further, B denotes the efective bandwidth, defined as the product of the allocated bandwidth and the fixed channel bandwidth $B _ { f }$ . Standard values for the gains were used as mentioned in Table V [46].

$$
S N R = P _ { T X } - P L + G _ { T X } - N P
$$

$$
D = B \log _ { 2 } ( 1 + S N R )\tag{2}
$$

(3)

The proposed resource management problem is formalized as a minimization problem subject to constraints hereafter. It is assumed that the resources $a = ( c _ { a } , m _ { a } , b _ { a } )$ is assigned for each demand $d = ( c _ { d } , m _ { d } , b _ { d } )$ . At a given time $t ,$ the diference between the allocated and demanded resources should be minimal provided the server capacities are not exceeded. Constraints C1 and C2 monitor the allocations by keeping them under the maximum capacity $C$ and more than the demand d. The measured latency L and data rate D thresholds are controlled by C3 and C4. Finally, in C5, the allocation of requests to the MEC servers is managed. All the incoming resource requests N with CPU, memory or bandwidth requirements are roughly divided as $x _ { i j }$ between the available M edge servers.

$$
\mathrm { m i n } | a _ { t } - d _ { t } |
$$

s.t.

$$
\begin{array} { r l r } & { \mathrm { C 1 : ~ } a _ { r } \leq C _ { r } } & { \mathrm { C 2 : ~ } a _ { r } \geq d _ { r } \quad \forall r \in \mathcal { R } } \\ & { \mathrm { C 3 : ~ } L \leq \mathcal { L } } & { \mathrm { C 4 : ~ } D \geq \mathcal { D } } \end{array}\tag{4}
$$

$$
\mathrm { C } 5 \colon \left| \sum _ { i } x _ { i j } - \frac { \mathbb { N } } { \mathbb { M } } \right| \le 1 \qquad \forall j
$$

TABLE IV: Summary of symbols
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $n _ { i }$ </td><td>MEC Servers</td></tr><tr><td> $c$ </td><td>CPU request</td></tr><tr><td> $m$ </td><td>Memory request</td></tr><tr><td> $^ { b }$ </td><td>Bandwidth request</td></tr><tr><td> $L$ </td><td>Latency</td></tr><tr><td> $\mathcal { L }$ </td><td>Latency limit</td></tr><tr><td> $D$ </td><td>Data rate</td></tr><tr><td> $\mathcal { D }$ </td><td>Data rate limit</td></tr><tr><td> $B$ </td><td>Effective bandwidth</td></tr><tr><td> $P _ { t x }$ </td><td>Transmit power</td></tr><tr><td> $P L$ </td><td>Path loss</td></tr><tr><td> $G _ { t x }$ </td><td>Antenna gain</td></tr><tr><td> $N F$ </td><td>Noise figure</td></tr><tr><td> $^ { a }$ </td><td>Resource allocations</td></tr><tr><td> $d$ </td><td>Resource demand</td></tr><tr><td>x</td><td>Server allocation</td></tr><tr><td>M</td><td>Total available edge servers</td></tr><tr><td>N</td><td>Total incoming resource requests</td></tr><tr><td> $Q$ </td><td>Query in the context of transformer</td></tr><tr><td> $K$ </td><td>Key in the context of transformer</td></tr><tr><td> $V$ </td><td>Value in the context of transformer</td></tr><tr><td> $W$ </td><td>Parameter matrix in the context of transformer</td></tr><tr><td> $\hat { r } _ { \psi } \sim \hat { R } _ { \psi }$ </td><td>Reward from MathΣtral model</td></tr><tr><td> $\hat { r } _ { \mu } \sim \hat { R } _ { \mu }$ </td><td>Reward from custom model</td></tr><tr><td> $r _ { \tau }$ </td><td>Instantaneous reward</td></tr><tr><td>R</td><td>Reward from individual agent</td></tr><tr><td> $E$ </td><td>Episode</td></tr><tr><td> $J$ </td><td>MASAC objective function</td></tr></table>

## B. Foundations for reasoning using LLM

The transformer architecture within LLMs allow them to translate the intent behind the information provided. Long-range dependencies between the sequence of words is established using self-attention mechanism which in turn is used for the reasoning [47]. Three main components are the query $Q .$ the key K and the value V. Consider a sequence of token embeddings $\boldsymbol { w } = [ w _ { 1 } , w _ { 2 } , w _ { 3 } , . . . . , w _ { n } ]$ for a sequence of length $n .$ The dimensionality of the queries and keys is $d _ { k }$ while that of the values is $d _ { v }$ . Additionally, consider the linear projections of the embeddings using learned parameter metrices $W ^ { Q } , W ^ { K } , W ^ { V }$ with respect to queries, keys and values. A softmax layer is applied to scaled dot-product of queries, keys and values produce the attention weights as in (5).

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right) V\tag{5}
$$

The single attention mechanism is further improved by computing multiple projections $( W _ { Q } ^ { ( i ) } , W _ { K } ^ { ( i ) } , \dot { W } _ { V } ^ { ( i ) } )$ from diferent sub-spaces to enhance information representation. In transformers, multiple independent h attention functions are used as multi-head attention mechanism as in (6) and (7).

TABLE V: System model parameters
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1> $\overline { { B _ { f } } }$ </td><td rowspan=1 colspan=1> $\overline { { N F } }$ </td><td rowspan=1 colspan=1> $\overline { { P _ { T X } } }$ </td><td rowspan=1 colspan=1>PL</td><td rowspan=1 colspan=1>GTX</td></tr><tr><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>20 MHz</td><td rowspan=1 colspan=1>7dB</td><td rowspan=1 colspan=1>46 dBm</td><td rowspan=1 colspan=1>125 dB</td><td rowspan=1 colspan=1>5.5 dBi</td></tr></table>

$$
\mathrm { h e a d } ^ { i } = \mathrm { A t t e n t i o n } ( Q ^ { i } , K ^ { i } , V ^ { i } )\tag{6}
$$

$$
\mathrm { M u l t i H e a d ( Q , K , V ) } = \mathrm { C o n c a t ( h e a d ^ { 1 } , . . . . , h e a d ^ { \it h } ) W \it { O } }\tag{7}
$$

## IV. RAGAM-AI design

This section navigates through the particulars of the conducted simulations. The section begins by defining the components of the Markovian Decision Process (MDP) formulation used by the RAGAM-AI algorithm. The experimental setup with the parameters that were utilized is further presented alongside the overall implementation process.

## A. Markovian approach

The decision-making within this work uses a Markovian approach as most of the RL algorithms do. The system in consideration is memoryless depending solely on the current state. Therefore, the MDP represented as $( S , { \mathcal { A } } , { \mathcal { P } } , { \mathcal { R } } , \gamma )$ is elaborated further as follows:

1) State: The state space is a set of diferent resource allocations and demands alongside the performance metrics latency and data rate. In (8) a and d are the allocations and demands respectively for each agent n ϵ N.

$$
{ \mathcal { S } } = \{ c _ { a } ^ { n } , m _ { a } ^ { n } , b _ { a } ^ { n } , c _ { d } ^ { n } , m _ { d } ^ { n } , b _ { d } ^ { n } , L ^ { n } , D ^ { n } \} \ n \epsilon \mathrm { N }\tag{8}
$$

2) Actions: The action space is a set of resource allocations $a _ { r e s o u r c e }$ predicted by the SAC policy given the observation from the environment for each agent.

$$
\mathcal { A } = \{ a _ { C P U } ^ { n } , a _ { m e m } ^ { n } , a _ { b w } ^ { n } \} ~ n \epsilon \mathrm { N }\tag{9}
$$

3) Reward function: The innovation for this work lies within the reward function of the formulation. The reward is designed using two sub-parts. In the first half to stabilize early training, the reward $r _ { \tau }$ is obtained by alternating between a custom reward function $\hat { R } _ { \mu }$ and the MathΣtral model ${ \hat { R } } _ { \psi }$ as in (10). $\hat { R } _ { \psi }$ is developed by querying using states $( s _ { 0 } , s _ { 1 } , . . . , s _ { t } )$ within CoT style prompting as in Fig.2 to obtain the instantaneous reward $\hat { r } _ { \psi }$ . The states provide values which are passed within the prompt instructions in a structured manner as visible in Fig.2 for going through series of constraint checks before being evaluated by the MathΣtral model. On the other hand, $\hat { r } _ { \mu }$ is the instantaneous custom reward sampled from $\hat { R } _ { \mu }$ that awards (w) or penalizes (λ) the constraints f(k) based on the satisfaction $\left( \mathbb { I } _ { ( \Omega \leq 0 ) } \right)$ . The parameters (w) and (λ) are empirically chosen by heuristics used to regulate the trade-of across multiple constraints while f(k) quantifies the deviation associated with constraint k. The sum of the awards and penalties is further adjusted for considering individual contribution R from each agent as seen in (11). The probability of using ${ \hat { R } } _ { \psi }$ or $\hat { R } _ { \mu }$ is linked to $p _ { \psi } ( e )$ , a stepwise decaying function dependent on steps e. The decaying function is added to gradually transition from the rewards generated by the LLM model to the surrogate model. During early training episodes, the framework queries the LLM more frequently while decaying exponentially to allow the surrogate model approximation for later.

$$
r _ { \tau } = \left\{ \begin{array} { l l } { \hat { r } _ { \psi } \sim \hat { R } _ { \psi } , } & { p _ { \psi } ( e ) } \\ { \hat { r } _ { \mu } \sim \hat { R } _ { \mu } , } & { 1 - p _ { \psi } ( e ) } \end{array} \right.\tag{10}
$$

$$
\hat { R } _ { \mu } = \frac { 1 } { \mathbb { R } } \sum _ { i = 0 } ^ { T } \left\{ w f ( k ) , \quad \mathrm { i f } \mathbb { I } _ { ( \Omega \leq 0 ) } \right.\tag{11}
$$

After collecting suficient samples, in the second half a surrogate DNN model is trained to give the reward score $\hat { r } _ { \tau }$ . The initial supervised dataset is provided by the statereward pairs from LLM inferencing. Later, a small DNN architecture is trained to approximate the LLM-generated state-reward pairs. Further, the surrogate model is put to use only after it is trained enough to have reasonably low Mean Absolute Error (MAE). In (12), $y _ { i }$ is the true value and $\hat { y } _ { i }$ is the predicted value from the surrogate model trained over $\mathcal { N }$ samples. Additionally, the MAE is subjected to a threshold ϵ to maintain the average prediction error between the rewards predicted by the diferent models to be under 10%.

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| y _ { i } - \hat { y } _ { i } \right|\tag{12}
$$

## B. Simulation setup

Fig.3 depicts the architecture behind the proposed algorithm. A multi-agent actor-critic network is structured within a centralized learning but decentralized execution framework. Further, the architecture includes the reward function utilizing the MathΣtral model while training a DNN after enough samples are collected. Algorithm 1 explains the simulation process step-by-step. The LLM model-based reward function is wrapped inside the MASAC implementation for receiving feedback to build an Agentic-AI solution. Here, the principle of RLAIF as mentioned before is applied. The experimental setup is further explained in detail below.

1) Dataset: For training purposes, the resource request available from Microsoft Azure Packing traces is used [48]. The publicly available dataset captures workloads from virtual machines, containers and tasks containing essential information about CPU, memory and bandwidth. These are large-scale data collected under varying realistic operational conditions. The traces were first cleaned by removing incomplete records and entries with missing or invalid resource usage values. A consistent sorting was applied to make sense of the workload ids and timestamps. The necessary resources such as CPU, memory and Bandwidth utilization value was retrieved. Hence, for this work the packing traces after extensive pre-processing were used as the resource requests.

![](images/d6e5c87158c3ab09390b9fc360f2e777f8f1d2445640e3c374e4ddcdf9886eb3.jpg)  
(b) Prompt pipeline  
Fig. 2: CoT-based reward generation workflow

2) Training conditions: A MASAC policy is used for the resource management at the edge in this work. The training is further augmented by bringing Gen-AI components as discussed on multiple occasions. The values for all the parameters that were used are mentioned in Table VI. Each agent iterates over a set of episodes at the end of which the training weights are updated and the RL environment is reset as visible from algorithm 1. The objective $J ( \pi )$ of the policy is to maximize the long-term cumulative reward $r _ { t }$ while enforcing entropy regularization $\mathcal { H } ( \pi _ { i } ( \cdot \mid o _ { i , t } ) )$ as in (13).

$$
J ( \pi ) = \mathbb { E } _ { \tau \sim \rho _ { \pi } } \left[ \sum _ { { t = 0 } } ^ { \infty } \gamma ^ { t } \left( r _ { t } + \sum _ { i = 1 } ^ { N } \alpha _ { i } \mathcal { H } ( \pi _ { i } ( \cdot \mid o _ { i , t } ) ) \right) \right]\tag{13}
$$

3) Language model settings: As explained previously, the reward function utilizes the MathΣtral model for generating rewards using CoT prompting. These design choices are made strategically keeping in mind the requirements of the problem formulated in section III. MathΣtral is a specialized model for mathematical reasoning. It is fine-tuned specifically on mathematical data to generate reliable mathematical outputs with detailed explanatory solutions [49]. However, for this model to successfully generate the desired outputs, the prompts passed on to the model have to be immaculate. Therefore, given the sequential decision-making process of the problem in hand, a CoT based prompt is designed as in Fig.2. A step-by-step approach is more structured and thus avoids reasoning errors by validating each constraint in comparison to any other prompting styles. Additionally, the prompt is not user-driven but rather a fixed, system-controlled component of the RAGAM-AI pipeline.

## V. Results

This section discusses the baseline policies used for comparison against the performance of RAGAM-AI. The results are further elaborated in terms of resource allocation, performance costs and the capacity to adapt to scaling networks. For each plot, the results from the steps of each episode is averaged to observe performance over the training episodes. Further, the critic loss convergence from the two critic networks within the three considered RL agents is shown in Fig.4. Also, to provide evidence for reward stability on using LLM, fifty random states are subjected to multiple LLM calls and three of those observations are reported in Table VII. It was concluded that rewards were stable even after multiple runs and hence the LLM-based reward design is able to consistently output the correct rewards.

TABLE VI: Experimental Parameters
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>MASAC</td><td rowspan=1 colspan=1>Actor learning rateCritic learning rateHidden layersNeurons per layer</td><td rowspan=1 colspan=1>0.0010.0001864</td></tr><tr><td rowspan=1 colspan=1>MathΣtral</td><td rowspan=1 colspan=1>TemperatureToken</td><td rowspan=1 colspan=1>0.260</td></tr><tr><td rowspan=1 colspan=1>GPU</td><td rowspan=1 colspan=1>ModelCUDA VersionMemory</td><td rowspan=1 colspan=1>NVIDIA 600012.649140MiB</td></tr><tr><td rowspan=1 colspan=1>Simulation</td><td rowspan=1 colspan=1>EpisodesStepsOptimizerActivation functionDiscount factor (γ)Batch sizeNumber of DNN layersLayer sizeTemperature parameter (α)</td><td rowspan=1 colspan=1>25512AdamReLu0.90648640.5</td></tr><tr><td rowspan=1 colspan=1>Implementation</td><td rowspan=1 colspan=1>LanguageRL LibraryLLM Library</td><td rowspan=1 colspan=1>Python 3.11.14Tensorflow 2.20.0PyTorch 2.9.1</td></tr></table>

![](images/e31fe132df29bc7b7a35c2235e11f16175a8a6c6104c0dbffe69f9746d09eb82.jpg)  
Fig. 3: RAGAM-AI algorithm architecture

Algorithm 1: RAGAM-AI   
for agent i = 1, . . . , M do   
Initialize parameters: actor $\theta _ { i } ,$ critic $\phi _ { i } .$   
temperature $\alpha _ { i } ,$ target critic $\phi _ { i } ^ { \prime }  \phi _ { i }$   
Initialize replay bufer D   
for episode =1 to N do   
environment ← environment.reset()   
state ← initial state s   
for steps = 1 to P do   
Collect observation $o _ { i }$   
Sample action for each agent $a _ { i } \sim \pi _ { \theta _ { i } } ( \cdot \mid o _ { i } )$   
Function Step(s, a):   
Function $R ( s , a ) \colon$   
if step mod K then   
train reward model   
if MAE < threshold ϵ then   
r ← Score using reward model   
else   
Sample $u \sim \mathcal { U } ( 0 , 1 )$   
if $u < p _ { \psi } ( e )$ then   
$\lfloor \mathrm { ~ r ~ }  \hat { R } _ { \mu }$   
else   
Build CoT prompt based on s   
$\mathrm { ~ r ~ } \gets \hat { R } _ { \psi }$   
Store in reward cache   
return r   
Collect next state $s ^ { \prime }$   
Collect done signal   
return (s′, r, done)   
Store transition $( s , a , r , s ^ { \prime } )$ in $\textit { \textbf { D } } s \gets s ^ { \prime }$   
if training step then   
Sample a minibatch from D for all   
agents   
Update parameters θ<sub>i</sub>, ϕ<sub>i</sub>, α<sub>i</sub>, ϕ′   
$s \gets s ^ { \prime }$   
Log values   
Store model weights

TABLE VII: Reward stability study
<table><tr><td colspan="3">Data</td></tr><tr><td colspan="3">Input States</td></tr><tr><td>Query runs</td><td colspan="2">50 5</td></tr><tr><td colspan="3">Reward stability</td></tr><tr><td>Mean</td><td>Standard deviation</td><td>Confidence</td></tr><tr><td>State 1</td><td>0.88</td><td>±0.07</td></tr><tr><td>State 2</td><td>0.06 0.68 0.08</td><td>±0.1</td></tr><tr><td>State 3</td><td>0.73 0.06</td><td>±0.07</td></tr></table>

## A. Baselines

The performance of the RAGAM-AI Algorithm is compared with a range of baselines with diferent complexities. All baselines are evaluated under the same experimental conditions using the same dataset, parameters and evaluation metrics to ensure a fair comparison. The following methods were implemented for benchmarking:

SAC: A single agent SAC algorithm with a traditional reward function (11) is used to highlight the performance without the support of Gen-AI components.

REINFORCE: The REINFORCE approach optimizes the policy parameters without learning the value function. It utilizes Monte-Carlo sampling for computing the estimates of policy gradients as in (14) while increasing the probability of actions that yield higher returns [50]. This approach is included to highlight the performance of a fairly less complex model-free policy in terms of learning stability while compared to an advanced RL algorithm.

$$
\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \pi _ { \theta } } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } ) G _ { t } \right]\tag{14}
$$

Random: In this approach, resources are assigned using uniform random sampling with respect to the constraints. A plausible set of allocations are selected without exceeding the server capacities.

## B. Resource allocation

Resource allocation by RAGAM-AI is evaluated by measuring the diference between the demand d and allocations a normalized to the total capacity. Fig.5 showcases the deviations achieved while using CPU, memory and bandwidth resources. The proposed RAGAM-AI exhibits improved performance in resource allocation by consistently reducing the deviation between the allocated resources and the observed demand. There is a consistent performance improvement by RAGAM-AI relative to baseline policies SAC, REINFORCE and random as reported in Table VIII. The baseline policies achieve higher deviations and costs leading to poor performance. While Table VIII provides the gain on comparing the mean values of the deviations, Fig.5 provides a visual representation of the overall deviation range. The results ascertain the fact that RAGAM-AI is able to take advantage of the hybrid nature of the reward function by successfully guiding the MASAC policy towards making eficient resource allocation decisions. Further, it can be established that using the occasional feedback from the LLM has introduced contextual reasoning enhancing the reward signals. The entropy-regulated SAC architecture when combined with LLM-assisted reasoning reliably maps resource supply and system demand enabling in a robust performance by RAGAM-AI.

![](images/320be5031011f7eebf242f0486b483d73b8a39d7a91f23a2b6fb46e1e97345d6.jpg)  
Fig. 4: Training loss

TABLE VIII: Performance gain relative to baselines
<table><tr><td>Metric</td><td>SAC</td><td>REINFORCE</td><td>Random</td></tr><tr><td>CPU deviation</td><td>17.33%</td><td>27.29%</td><td>47.36%</td></tr><tr><td>Memory deviation</td><td>16.38%</td><td>17.50%</td><td>56.09%</td></tr><tr><td>Bandwidth deviation</td><td>19.27%</td><td>23.01%</td><td>56.52%</td></tr><tr><td>Latency cost</td><td>2e-4</td><td>2.4e-4</td><td>1.42e-3</td></tr><tr><td>Data rate cost</td><td>2e+6</td><td>3e+6</td><td>7e+6</td></tr></table>

## C. Performance cost

In addition to demand-allocation deviation, performance cost is evaluated in terms of latency and data rate cost. These metrics are weighed in by utilizing the allocated and total bandwidth to include budget-aware eficient resource allocations. Further, the performance metrics should be evaluated relative to the amount of resources consumed and prevents bias toward larger allocations. It can be visualized in Fig.6 that RAGAM-AI yields the least performance cost in terms of both latency and data rate in comparison to the considered baselines. Similar to the resource allocation, a cost improvement can be observed from Table VIII. Enhanced reward shaping utilizing the exploration from the MASAC policy and reasoning capabilities of MathΣtral capture the multiobjective constraints of the system better than the other algorithms. Further, RL optimizes long-term decision policies through interaction, while the LLM provides contextual priors leading to more eficient resource utilization under budget constraints.

![](images/f2ef4a0b00db3a056cd501fb2d22dc141a516d86e55b9c2cde8b420ae6e6bac8.jpg)  
(a) CPU deviation

![](images/f808b1b34c902822fd79c0122505de1cdd6ea7644d4be163002dd87a61cb1a0d.jpg)  
(b) Memory deviation

![](images/aa82a3d5a3912d21aee021527e6fee0a38f5abffccacaecd6434c52c88ac1d61.jpg)  
(c) Bandwidth deviation  
Fig. 5: Resource allocation performance

![](images/61e7735ec09876e6a1ad6e58cc30eecc1970bed24cf595ba104edcc4eab9a6b9.jpg)

(a) Latency cost  
![](images/a80695fb37293be35d2516ec98a92b2cbfec00b58400f8809971402eb17b72fa.jpg)  
(b) Data rate cost  
Fig. 6: Performance costs

## D. Adaptability

In this work, the multi-agent RAGAM-AI based on RL-LLM framework with a systematic reward mechanism contributes significantly to system scalability and adaptability. The decentralized learning component within the RL algorithm enables each individual agent to learn policies that optimize local decision-making while adhering overall system objectives designed within the reward function. It is to be noted that the reward is not directly optimizing the resource deviations or KPI costs, rather it allows the framework to learn the network demand patterns while keeping the constraints in check. With the assistance from

LLM, agents are able to interpret contextual information and introduce feedback using its pre-trained knowledge. The hybrid approach within the reward function further helps in the development of a surrogate reward model that acts a substitute replacing the otherwise undetermined task. Although the proposed RAGAM-AI method exhibits a larger spread in Fig. 5 and Fig. 6, it can be inferred as a consequence of enhanced exploration and variability in the learned policy during training. Despite this variability, the RAGAM-AI method consistently achieves lower median cost, indicating superior performance compared to baseline approaches. The agentic system thus built iteratively adjusts its actions dynamically to meet the heterogeneous resource demands to maximize the performance. Overall, the system complexity is addressed with respect to two components: the MADRL setup and the LLM-based reward mechanism. Neither of the components require continuous inferencing on resource-constrained devices. Further, the surrogate reward model alleviates the complexity issues by enabling lightweight decisionmaking. Therefore, the proposed RAGAM-AI framework is more suitable for edge controllers or orchestration nodes in terms of complexity.

## VI. Conclusions

To summarize, in this work a hybrid Agentic-AI framework RAGAM-AI is implemented by combining the abilities of RL and Gen-AI. The multi-agent system was developed using the MASAC policy while receiving feed backs from the MathΣstral inside its reward function. Network resources such as CPU, memory and bandwidth are managed at the network edge using RAGAM-AI. The main objective was to strengthen the performance of the MADRL system by augmenting it with context-aware reasoning. Hence, RAGAM-AI demonstrates efectiveness of RLAIF concept by obtaining smaller resource deviations and performance costs in comparison to the baselines. For future work, a workflow encompassing diferent agents conveying intents from diferent components such as Open-Radio Access Network (O-RAN) and cloud can be integrated. Further, this work serves as simulation for the proof-of-concept idea proposed and thus can be tested by deploying in cloud-native environments as well later. Broader metrics and processes such as caching may also be considered. Including system-level analysis in terms of runtime overhead measurements, convergence analysis, and sensitivity studies can also provide important additional insights for reducing the system complexity.

## References

[1] P. Peng et al.,“A survey on computation ofloading in edge systems: From the perspective of deep reinforcement learning approaches,” Computer Science Review, vol. 53, 2024.

[2] F. Golpayegani et al., “Adaptation in Edge Computing: A Review on Design Principles and Research Challenges,” ACM Transactions on Autonomous and Adaptive Systems, vol. 19, no. 13, September 2024.

[3] S. Dong et al., “Task ofloading strategies for mobile edge computing: A survey,” vol. 254, December 2024.

[4] X. Zhang et al., “A Survey on Privacy-Preserving Caching at Network Edge: Classification, Solutions, and Challenges,” ACM Computing Surveys, vol. 57, no. 5, May 2025.

[5] P. Arroba et al., “Sustainable edge computing: Challenges and future directions,” Journal of Software: Practice and Experience, vol. 54, no. 11, pp. 2272-2296, November 2024.

[6] M. Ergen et al., “Edge computing in future wireless networks: A comprehensive evaluation and vision for 6G and beyond,” ICT Express, vol. 10, no. 5, pp. 1151-1173, October 2024.

[7] H. Seo et al., “Kafka-Thor: A Kafka-based In-Edge Data Streaming Platform for Enhanced V2X Services,” IEEE Transactions on Services Computing.

[8] B. Dande et al., “An Integrated Charging and Computation Scheduling of Electric Vehicles in Edge Computing System,” IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 1, pp. 916-934, January 2025.

[9] F. Zhang et al., “Computing Sandbox Driven Secure Edge Computing System for Industrial IoT,” IEEE Transactions on Network and Service Management, vol. 22, no. 5, pp. 4538-4550, October 2025.

[10] A. Duru et al., “Resource Allocation for XR with Edge Offloading: A Reinforcement Learning Approach,” in Proc. IEEE Military Communications Conference (MILCOM), Los Angeles, CA, USA, 2025.

[11] X. Chen et al., “Mobility-Aware Dependent Task Ofloading in Edge Computing: A Digital Twin-Assisted Reinforcement Learning Approach,” EEE Transactions on Mobile Computing, vol. 24, no. 4, pp. 2979-2994, April 2025.

[12] T. Jiang et al., “Deep-Reinforcement-Learning-Based Task Offloading and Resource Allocation in Mobile Edge Computing Network With Heterogeneous Tasks,” IEEE Internet of Things Journal, vol. 12, no. 8, pp. 10899-10906, April 2025.

[13] X. Zhao et al., “MAPSM: Mobility-Aware Proactive Service Migration Framework for Mobile-Edge Computing in Consumer Internet of Vehicles,” IEEE Transactions on Consumer Electronics, vol. 71, no. 2, pp. 3753-3766, May 2025.

[14] S. Deng et al., “Privacy Preservation in User Behavior Analysis for Mobile-Edge Computing,” IEEE Transactions on Consumer Electronics, vol. 71, no. 2, pp. 3718-3730, May 2025.

[15] J. Sasiain et al., “Toward the Integration and Convergence Between 5G and TSN Technologies and Architectures for Industrial Communications: A Survey,” IEEE Communications Surveys & Tutorials, vol. 27, no. 1, pp. 259-321, February 2025.

[16] S. Jiang et al., “Reinforcement-Learning-Based Network Slicing and Resource Allocation for Multi-Access Edge Computing Networks,” IEEE Transactions on Cognitive Communications and Networking, vol. 10, no. 3, pp. 1132-1145, June 2024.

[17] J. Tan et al., “Multi-Objective Optimization Using Adaptive Distributed Reinforcement Learning,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 9, pp. 10777- 10789, September 2024.

[18] S. K. Chari et al., “Cooperative Multi-Agent Deep Reinforcement Learning for Service-Aware Energy Optimization at the Network Edge,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 4124-4137, 2026.

[19] S. Khan et al., “EMDTORA: Energy-Aware Multi-User Dependent Task Ofloading and Resource Allocation in MEC Using Graph-Enabled DRL,” in IEEE Transactions on Green Communications and Networking, vol. 9, no. 3, pp. 1453-1469, September 2025.

[20] Y. Xu et al., “Mixture of Experts for Decentralized Generative AI and Reinforcement Learning in Wireless Networks: A Comprehensive Survey,” IEEE Communications Surveys & Tutorials.

[21] X. Qin et al., “Generative AI Meets Wireless Networking: An Interactive Paradigm for Intent-Driven Communications,” in IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 4, pp. 2056-2077, August 2025.

[22] K.M. Lo et al., “A Closer Look into Mixture-of-Experts in Large Language Models,” Association for Computational Linguistics, pp. 4427-4447, April 2025.

[23] S. Hao et al., “Online Learning from Strategic Human Feedback in LLM Fine-Tuning,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1-5, 2025.

[24] Y. Bai et al., “Constitutional AI: Harmlessness from AI feedback,” 2022, arXiv:2212.08073.

[25] T. Kwa et al., “Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification,” in Proc. Neural Information Processing Systems (NeurIps), pp. 14608-14633, vol. 37, 2024.

[26] C. Laidlaw et al., “Preventing Reward Hacking with Occupancy Measure Regularization,” in Proc. International Conference on Learning Representations (ICLR), 2024.

[27] S.K. Chari et al., “Generative AI-Augmented Reinforcement Learning for Enhanced Edge Resource Optimization,” IEEE Global Communications Conference, December 2025, Taiwan.

[28] C. Yu et al., “A Multi-Objective Evolutionary Approach: Task Ofloading and Resource Allocation Using Enhanced Decomposition-Based Algorithm in Mobile Edge Computing,” IEEE Access, vol. 12, pp. 123640-123655, 2024.

[29] Y. Zhang et al., “Joint Trajectory and Resource Optimization for UAV and D2D-Enabled Heterogeneous Edge Computing Networks,” IEEE Transactions on Vehicular Technology, vol. 73, no. 9, pp. 13816-13827, September 2024.

[30] W. Chu et al., “Joint Service Caching, Resource Allocation and Task Ofloading for MEC-Based Networks: A Multi-Layer Optimization Approach,” IEEE Transactions on Mobile Computing, vol. 23, no. 4, pp. 2958-2975, April 2024.

[31] J. Zeng et al., “Resource-Eficient Joint Service Caching and Workload Scheduling in Ultra-Dense MEC Networks: An Online Approach,” IEEE Transactions on Network and Service Management, vol. 22, no. 2, pp. 1788-1800, April 2025.

[32] M. Elkawkagy et al., “Genetic Algorithm-Driven Joint Optimization of Task Ofloading and Resource Allocation for Fairness-Aware Latency Minimization in Mobile Edge Computing,” IEEE Access, vol. 13, pp. 118237-118248, 2025.

[33] L. Wang et al., “Eficient Deep Reinforcement Learning-Based Resource Allocation for Cloud Native Wireless Network,” IEEE Transactions on Green Communications and Networking, vol. 9, no. 4, pp. 2236-2248, December 2025.

[34] W. Zhang et al., “Deep-Reinforcement-Learning-Based Joint Caching and Resources Allocation for Cooperative MEC,” IEEE Internet of Things Journal, vol. 11, no. 7, pp. 12203-12215, April 2024.

[35] Y. Dai, “Federated Deep Reinforcement Learning for Task Ofloading in Digital Twin Edge Networks,” IEEE Transactions on Network Science and Engineering, vol. 11, no. 3, pp. 2849- 2863, 2024.

[36] K. Yang et al., “Knowledge-Defined Edge Computing Networks Assisted Long-Term Optimization of Computation Ofloading and Resource Allocation Strategy,” IEEE Transactions on Wireless Communications, vol. 23, no. 6, pp. 5316-5329, June 2024.

[37] Z. Liu et al., “DNN Partitioning, Task Ofloading, and Resource Allocation in Dynamic Vehicular Networks: A Lyapunov-Guided Difusion-Based Reinforcement Learning Approach,” IEEE Transactions on Mobile Computing, vol. 24, no. 3, pp. 1945-1962, March 2025.

[38] D. C. Nguyen et al., “Cooperative Task Ofloading and Block Mining in Blockchain-Based Edge Computing With Multi-Agent Deep Reinforcement Learning,” IEEE Transactions on Mobile Computing, vol. 22, no. 4, pp. 2021-2037, April 2023.

[39] S. Cheng et al., “Resource-Aware Reinforcement Learning-Based Transmission Optimization for Mobile Augmented Reality in Edge Computing,” IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 4, pp. 2356-2367, August 2025.

[40] C. Huang et al., “Joint Ofloading and Resource Allocation for Hybrid Cloud and Edge Computing in SAGINs: A Decision Assisted Hybrid Action Space Deep Reinforcement Learning Approach,” IEEE Journal on Selected Areas in Communications, vol. 42, no. 5, pp. 1029-1043, May 2024.

[41] X. Zheng et al., “EdgeNetLLM: Cloud–Edge Collaborative Adaptation of Large Language Models for Mobile Networking,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 3928-3943, 2026.

[42] Z. Liu et al., “Two-Timescale Model Caching and Resource Allocation for Edge-Enabled AI-Generated Content Services,” IEEE Transactions on Mobile Computing.

[43] X. Tang et al., “LLM-Assisted Reinforcement Learning: Leveraging Lightweight Large Language Model Capabilities for Efficient Task Scheduling in Multi-Cloud Environment,” IEEE Transactions on Consumer Electronics, vol. 71, no. 2, pp. 5631- 5644, May 2025.

[44] H. Du et al., “Difusion-Based Reinforcement Learning for Edge-Enabled AI-Generated Content Services,” IEEE Transactions on Mobile Computing, vol. 23, no. 9, pp. 8902-8918, September 2024.

[45] C. E. Shannon, “A mathematical theory of communication,” 7 The Bell System Technical Journal, vol. 27, no. 3, pp. 379–423, 1948.

[46] ETSI, “Study on International Mobile Telecommunications (IMT) parameters for 6.425 - 7.025 GHz, 7.025 - 7.125 GHz and 10.0 - 10.5 GHz,” 3GPP TR 38.921 version 17.1.0, 2022.

[47] A. Vaswani et al., “Attention is all you need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[48] Ori Hadary et al., “Protean: VM Allocation Service at Scale,” in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI 2020), November 2020.

[49] Mathstral. [Online]. Accessed: February 27, 2026. Available:https://mistral.ai/news/mathstral

[50] R.J. Williams et al., “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,” Machine Learning, Vol. 8, no. 3-4, pp. 229-256, 1992.