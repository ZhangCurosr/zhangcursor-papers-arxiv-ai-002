# User-Assisted Collaborative Distributed Inference for Eficient QoS-Aware Autoscaling

Alfreds Lapkovskis , Ali Beikmohammadi , Sindri Magn´usson , and Praveen Kumar Donta

Department of Computer Systems and Sciences, Stockholm University SE-106 91 Stockholm, Sweden

{alfreds.lapkovskis, beikmohammadi, sindri.magnusson, praveen}@dsv.su.se

Abstract. Growing demand for artificial intelligence (AI) inference services requires scalable infrastructure, yet centralized serving costs rise with demand. We propose a collaborative distributed inference system combining dedicated infrastructure with resources contributed by service users. Dedicated resources provide baseline capacity for maintaining quality of service (QoS), while volunteered resources absorb increasing demand without proportional growth in centralized infrastructure. To capture stochastic and dynamic interactions among users, resources, tasks, and policies, we develop a high-dimensional generative Markov model with structured temporal factorization. The model supports simulation and provides a foundation for task scheduling and QoS-aware resource allocation optimization. We evaluate the system across user populations, resource capacities, and centralized and distributed scheduling policies. Simulations show that distributed scheduling becomes increasingly advantageous as the user population grows, improving request completion and P99 latency while substantially reducing dedicated resource consumption. These results demonstrate the feasibility of user-assisted collaborative inference for infrastructure-eficient autoscaling.

Keywords: Distributed AI Inference · AI Inference Services · Generative Markov Model · Quality of Service · QoS-Aware Scheduling · Resource Autoscaling · Volunteer Computing

## 1 Introduction

Artificial intelligence (AI) inference has become a core service in modern computing systems. According to the AI Index Report 2026, corporate AI adoption has reached 88%, while global investments surged to \$581.69 billion in 2025 [21]. As AI inference scales to meet this demand, cumulative inference costs increasingly outpace one-time training expenses, making operational eficiency a growing concern. These trends highlight the importance of the computing paradigms underlying AI inference. Central to these paradigms is cloud computing, the de facto standard for Internet applications [18], including AI inference. Its key advantages include on-demand resource provisioning, rapid elasticity, standardized access interfaces, utilization monitoring, and pay-as-you-go pricing [17,15].

However, centralized clouds are also subject to significant challenges. The costs associated with the setup and management of cloud infrastructure are prohibitively high [19,18]. Furthermore, expenses grow proportionally with demand, making cloud computing increasingly costly at scale. This scaling also carries a substantial environmental burden. Training Grok 4 alone emitted more CO than an average car produces over its entire lifetime, and the annual inference cost of GPT-4o exceeds the drinking water demand of 12 million people [21]. More broadly, cloud data centers consume vast amounts of energy and freshwater, with AI workloads emerging as a primary driver [13], and these impacts are projected to worsen as AI demand continues to rise.

Edge and fog computing partially address these limitations. By distributing computation closer to end users and data sources, they reduce latency and bandwidth consumption, and decrease reliance on centralized cloud resources [7]. The computing continuum extends this approach by integrating cloud, fog, and edge tiers into a unified, synergistic architecture [8]. Recent work has improved inference within these environments through heterogeneous on-device coexecution [26], task-aware DNN partitioning in mobile edge computing [9], and joint batching and transformer partitioning for distributed LLM inference [28]. Other studies address on-device inference techniques [24], disaggregated inference deployment across heterogeneous infrastructure [27], inference-data distribution for model monitoring [16], and cloud inference autoscaling [11]. These approaches improve inference within predefined deployment environments, but generally rely on dedicated infrastructure rather than resources that become available dynamically as users join the service. Moreover, computing-continuum systems still face architectural and economic barriers to widespread adoption [6,20].

Yet deploying dedicated edge and fog infrastructure is not the only path forward. A vast number of commodity devices owned by individuals and institutions worldwide, remain largely underutilized [4,19]. Harnessing these idle resources toward a shared computational objective is known as volunteer computing [15,18], a paradigm that ofers access to cheaper, distributed compute without requiring energy-hungry data center infrastructure. However, existing systems provide limited support for stable QoS guarantees due to resource heterogeneity and intermittent availability [18]. Hence, this model has limited applicability to consumer-oriented AI inference applications. Yet the underlying concept remains compelling in the context of AI inference, where operational costs are a growing concern.

Therefore, we propose a collaborative distributed inference system that partitions inference computations between dedicated resources, such as centralized servers, and volunteered resources, such as users’ devices (as shown in Fig. 1). Unlike centralized cloud inference or approaches distributed exclusively across dedicated computing continuum infrastructure, our system does not require dedicated computational capacity to scale in proportion to demand. In contrast to conventional volunteer computing, however, it retains dedicated infrastructure to provide a reliable baseline capacity and support QoS guarantees despite the intermittent availability and heterogeneity of volunteered resources. Given an appropriate scheduler, the server can process all requests when few users are online; as the number of available users grows, the demand increases, but so does the pool of volunteered resources, allowing the system to progressively offload computation to user devices. Determining how to schedule tasks across these dynamic resources and how much dedicated capacity is necessary to satisfy QoS requirements requires a model that captures the complex interactions among users, resources, tasks, and policies. We therefore develop such a model to enable system analysis, scheduling optimization, and QoS-aware allocation of dedicated resources. Our contributions are threefold:

![](images/734367b86459ef4af493466317476ee7d97befb56072032b3aca235bd810245b.jpg)  
Fig. 1: Overview of the proposed distributed inference system. The server receives requests, schedules subtasks across dedicated and volunteered resources according to the scheduling policy, and aggregates the results into responses.

1. We propose a novel collaborative distributed AI inference system that combines the dedicated and volunteered user resources for automatic scaling while maintaining QoS and reducing the need for centralized resources.

2. We model the system as a generative Markov model whose temporal factorization captures sparse dependencies among structured state variables and their elements, enabling tractable simulation, inference, and policy learning.

3. We evaluate and compare various scheduling policies by simulating the proposed system under diverse configurations, varying the number of users, dedicated, and volunteered resources.

## 2 Proposed Work

We propose a collaborative distributed inference system that partitions inference computations between dedicated resources, such as centralized servers, and volunteered resources, such as users’ devices. Ensuring QoS in such a system requires not only provisioning additional dedicated resources but also optimizing computation scheduling strategies. In principle, QoS can be maintained either by increasing server capacity and executing more computations on dedicated infrastructure or by improving the scheduler to make more efective use of volunteered resources. However, volunteered resources are inherently heterogeneous, highly dynamic, and intermittently available. Scheduling decisions must therefore be adaptive and capable of anticipating future system states to prevent QoS degradation while eficiently utilizing the available resource pool. Optimizing such scheduling strategies directly on a live system is impractical: it would be costly, time-consuming, and require extensive trial-and-error exploration in a complex, dynamic environment. Consequently, a system model is needed to enable eficient evaluation and optimization of scheduling strategies. In the following section, we propose our model of the system.

## 2.1 System Model

We model the system using the generative Markov framework introduced in [12], where states $\mathbf { s } _ { t }$ evolve over time according to the Markov property. The framework provides a unified representation of system components and allows their distributions to be modeled, modified, and extended independently. Its sparse factorization makes the high-dimensional model tractable for simulation, inference, and policy learning. We consider a system consisting of a dedicated server $v _ { 0 } ,$ a set of users $i \in \mathcal { T } ,$ and a set of nodes $v \in V = \{ v _ { 0 } \} \cup T$ , each providing resources $r \in \mathcal { R }$ . The system supports types of inference tasks, where each task type $k \in \mathcal { K }$ consists of a set of subtasks $p \in \mathcal { P } _ { k }$ . Users can both request inference tasks and contribute resources to execute subtasks. The system is controlled by two decision-making policies: a scheduler policy π and an executor policy ς. Together, these policies generate the joint action ${ \mathbf { u } } _ { t } = \langle { \mathbf { u } } _ { t } ^ { \pi } , { \mathbf { u } } _ { t } ^ { \varsigma } \rangle$ . We assume a fixed interval of 1 s between timesteps t. We define the system state $\mathbf { s } _ { t }$ as a tuple of high-dimensional variables:

$$
\mathbf { s } _ { t } = \langle \mathbf { o } _ { t } , \mathbf { q } _ { t } , \mathbf { a } _ { t } , \mathbf { x } _ { t } , \mathbf { y } _ { t } , \mathbf { c } _ { t } , \mathbf { d } _ { t } , \mathbf { u } _ { t } \rangle ,\tag{1}
$$

where each variable captures a specific aspect of the system state. Each variable is indexed by the system entities to which it applies, with one dimension assigned to each entity type. Following the Markov property, variables may depend on other variables within the same or previous time step. These dependencies are generally sparse, since many entities are independent in various state variables. In the following sections, we describe the system variables and the implementations of the decision-making policies.

## 2.2 System Variables

Availability states $\left( \mathbf { o } _ { t } \right)$ . We represent user availability with the online indicator $\mathbf { o } _ { t } \in \{ 0 , 1 \} ^ { | \mathcal { T } | }$ and the state-duration variable $\mathbf { d } _ { t } ^ { o } \in \mathbb { N } ^ { | \mathcal { T } | }$ , where $o _ { t , i } = 1$ indicates that user $i \in \mathcal { T }$ is online and $d _ { t , i } ^ { o }$ denotes the number of time steps since the last online/ofline transition. These variables evolve as ${ \bf d } _ { t } ^ { o } \sim { \cal P } ( \cdot \mid { \bf d } _ { t - 1 } ^ { o } , { \bf o } _ { t - 1 } )$ and $\mathbf { o } _ { t } \sim P ( \cdot \mid \mathbf { o } _ { t - 1 } , \mathbf { d } _ { t } ^ { o } )$ , factorized over users under independent availability processes. The duration dependence captures the fact that transition probabilities change with the time spent in the current state, while $\mathbf { o } _ { t - 1 }$ selects the corresponding online or ofline duration model. For each user, $d _ { t , i } ^ { o }$ increments if the state persists and resets to 0 if it changes; $o _ { t , i }$ then remains equal to $O _ { t - 1 , i }$ when $d _ { t , i } ^ { o } > 0$ and switches to $1 - o _ { t - 1 , i }$ otherwise.

We model availability durations with parametric survival distributions, following findings from [10], and sample transitions from their discrete-time hazards. We use a Weibull survival function $S ( t ) \triangleq \exp ( - ( t / \lambda ) ^ { k } )$ for online durations, with $\lambda = 1 1 3 3 . 1$ and $k = 0 . 5 7 6 4$ , and a log-normal survival function $S ( t ) \triangleq 1 - \Phi ( ( \log t - \mu ) / \sigma )$ for ofline durations, with $\mu = \log ( 5 2 5 9 ) , \sigma = 1 . 5 ,$ and standard normal CDF Φ( ). Given duration $d - 1$ , the current state persists with probability $S ( d ) / S ( d - 1 )$ and changes otherwise. These parameters yield median online and ofline durations of approximately 10 minutes and 1.5 hours, respectively, and a steady-state user availability of approximately 10%.

Inference requests $\left( \mathbf { q } _ { t } \right)$ . We represent inference requests with $\mathbf { q } _ { t } \in \mathcal { Q } ^ { \vert \boldsymbol { \tau } \vert }$ 2 where $\mathcal { Q } = \mathcal { K } \cup \{ k _ { 0 } \}$ and $k _ { 0 }$ denotes no active request, and with the request-state duration $\mathbf { d } _ { t } ^ { q } \in \tilde { \mathbb { N } } ^ { | \mathcal { I } | }$ , where $d _ { t , i } ^ { q }$ counts the time steps since the last request-state change of user i. The variables evolve as $\mathbf { d } _ { t } ^ { q } \sim P ( \cdot \mid \mathbf { d } _ { t - 1 } ^ { q } , \mathbf { q } _ { t - 1 } , \mathbf { o } _ { t } , \mathbf { y } _ { t - 1 } )$ and $\mathbf { q } _ { t } \sim P ( \cdot \mid \mathbf { q } _ { t - 1 } , \mathbf { o } _ { t } , \mathbf { d } _ { t } ^ { q } )$ , factorized over users. Users can submit requests only while online; hence, if $o _ { t , i } = 0$ , we set $q _ { t , i } = k _ { 0 }$ and $d _ { t , i } ^ { q } = 0$ . Each user has at most one active request, so requests alternate between k<sub>0</sub> and some $k \in \mathcal { K }$ . The duration $d _ { t , i } ^ { q }$ increments while the request state persists and resets when a request starts, an active request completes according to $\mathbf { y } _ { t - 1 }$ , or the user abandons it.

In the implementation, we use a Pareto survival function $S ( t ) \triangleq 1$ for $t < \kappa$ and $S ( t ) \triangleq ( \kappa / t ) ^ { \alpha }$ for $t \geq \kappa ,$ , following heavy-tailed inter-request times observed in interactive systems [2,5]. We use this distribution both for user think time, when $q _ { t - 1 , i } = k _ { 0 }$ , and patience, when $q _ { t - 1 , i } \neq$ k<sub>0</sub> and the request has not completed. Given duration $d - 1$ , the current request state persists with probability $S ( d ) / S ( d - 1 )$ and changes otherwise; when a new request starts, we sample its type k from $\mathrm { C a t e g o r i c a l } ( p _ { 1 } , \dots , p _ { | { \cal K } | } )$ . We set $\kappa = 3 0 \mathrm { s }$ and $\alpha = 1 . 5$ , yielding a minimum think time and patience of $\mathrm { 3 0 s , }$ a median of approximately $4 8 \mathrm { s } ,$ and equal steady-state fractions of time with and without an active request.

Allocated resources $\left( \mathbf { a } _ { t } \right)$ . We represent allocated resources with $\begin{array} { r l } { \mathbf { a } _ { t } } & { { } \in \mathbf { \pi } } \end{array}$ $\mathbb { R } _ { > 0 } ^ { | \mathcal { R } | \times | V | }$ , where $\boldsymbol { a } _ { t , r , v }$ denotes the amount of resource $r \in \mathcal { R }$ allocated to the system on node $v \in V$ . The variable evolves as $\mathbf { a } _ { t } \sim P ( \cdot \mid \tilde { \mathbf { a } } _ { t - 1 } , \mathbf { o } _ { t } )$ , where $\tilde { \mathbf { a } } _ { t - 1 } = [ \mathbf { a } _ { t - 1 } , \dots , \mathbf { a } _ { t - H } ]$ captures the autocorrelation of resource trajectories. We factorize the transition over resources and nodes, condition user resources on availability, and treat the server $v _ { 0 }$ as always online. Thus, we set $a _ { t , r , i } = 0$ for ofline users and sample allocations for online users and the server. Each $\boldsymbol { a } _ { t , r , v }$ represents the total resource capacity allocated to the system, including both used and unused capacity.

In the implementation, we use = band in, band out, cpu, memory, storage and fit one generative ARMA model [3] per resource. We collect workstation measurements every 3 s for 4 hours, resample them to ${ 1 \mathrm { s } } ,$ standardize each resource, and apply a Yeo–Johnson transformation [25]. For each $r ,$ we fit ARMA models with parameters $p \in \{ 1 , 2 , 5 , 1 0 \}$ and $q \in \{ 0 , 1 , 2 , 5 , 1 0 \}$ by maximum likelihood, selecting $\mathrm { A R M A } _ { * } ^ { A , r }$ with the lowest AIC [1]. During simulation, we sample from $\mathrm { A R M A } _ { * } ^ { \breve { A } , r }$ , invert the transformation, rescale using node-specific mean $\mu _ { r , \imath } ^ { A }$ and standard deviation $\sigma _ { r , v } ^ { A }$ , and clip to $[ \underline { { a } } _ { r , v } ^ { o } , \overline { { a } } _ { r , v } ^ { o } ]$ , where $\underline { { a } } _ { r , v } ^ { o } = 0$ and $\overline { { a } } _ { r , v } ^ { o }$ denote the minimum and maximum permitted allocations. We assume homogeneous user nodes but allow diferent server parameters.

Subtask readiness states $\left( \mathbf { x } _ { t } \right)$ . We represent subtask-data readiness with an auxiliary state $\hat { \mathbf { x } } _ { t } \in \mathcal { X } ^ { | V | \times | K | \times P _ { \operatorname* { m a x } } ^ { } }$ , a duration variable d $\mathbf { \Delta } _ { t } ^ { x } \in \mathbb { N } ^ { | V | \times | \mathcal { K } | \times P _ { \operatorname* { m a x } } }$ and the actual readiness state $\mathbf { x } _ { t } \in \mathcal { X } ^ { | V | \times | K | \times P _ { \operatorname* { m a x } } }$ , where $\mathcal { X } = \{ \mathsf { e m p t y }$ , download, <sup>pause,</sup> <sup>ready</sup>}<sup>,</sup> $P _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k \in \mathcal { K } } | \mathcal { P } _ { k } |$ , and $p \in \mathcal { P } _ { k }$ indexes subtasks of task type $k .$ These variables evolve as $\hat { \mathbf { x } } _ { t } \sim P ( \cdot \mid \mathbf { x } _ { t - 1 } , \mathbf { o } _ { t } , \mathbf { u } _ { t - 1 } ^ { x } ) , \mathbf { d } _ { t } ^ { x } \sim P ( \cdot \mid \mathbf { d } _ { t - 1 } ^ { x } , \hat { \mathbf { x } } _ { t } , \mathbf { c } _ { t } ^ { x } )$ ， and $\mathbf { x } _ { t } \sim P ( \cdot \mid \hat { \mathbf { x } } _ { t } , \mathbf { d } _ { t } ^ { x } )$ , factorized over nodes, task types, and subtasks. The auxiliary state first applies scheduler and executor actions $\mathbf { u } _ { t - 1 } ^ { x } = \langle \mathbf { u } _ { t - 1 } ^ { x , \pi } , \mathbf { u } _ { t - 1 } ^ { x , \varsigma } \rangle \colon$ the scheduler can start or cancel downloads, while the executor can pause, resume, or abort them. We suppress transitions for ofline users and treat the server $v _ { 0 }$ as always online; moreover, the server stores all subtask data a priori.

The duration $d _ { t , v , k , p } ^ { x }$ matters only when $\widehat { x } _ { t , v , k , p } =$ download. For all other states, we set it to $0 .$ . During an active download, $d _ { t , v , k , p } ^ { x }$ increments while the accumulated storage consumption $\boldsymbol { c } _ { t , \mathbf { s t o r a g e } , v , k , p } ^ { x }$ remains below the required subtask-data size $c _ { k , p }$ , and resets to 0 once $c _ { t , \mathrm { s t o r a g e } , v , k , p } ^ { x } = c _ { k , p }$ . The actual state $\mathbf { x } _ { t }$ copies $\hat { \mathbf { x } } _ { t }$ , except that $\hat { x } _ { t , v , k , p } =$ download becomes $x _ { t , v , k , p } =$ ready when $d _ { t , v , k , p } ^ { x } = 0$ , indicating download completion. In the implementation, paused downloads retain their accumulated storage, ready subtasks keep storage fixed at ${ \mathit { c } } _ { k , p } ,$ and all non-storage download consumptions vanish outside the download state.

Subtask execution states $\left( \mathbf { y } _ { t } \right)$ . We represent subtask execution with an auxiliary state $\hat { \mathbf { y } } _ { t } \in \mathcal { V } ^ { | \mathcal { T } | \times | V | \times | K | \times \tilde { P _ { \operatorname* { m a x } } } }$ , a duration variable $\mathbf { d } _ { t } ^ { y } \in \mathbb { N } ^ { | \mathcal { T } | \times | V | \times | K | \times P _ { \operatorname* { m a x } } }$ and the actual execution state $\mathbf { y } _ { t } ~ \in ~ \mathcal { y } ^ { | \mathbb { Z } | \times | V | \times | K | \times P _ { \operatorname* { m a x } } }$ , where $y \ = \ \{ \mathrm { i d l e } .$ download, execution, upload, done . The entry $y _ { t , i , v , k , p }$ describes the execution state of subtask $p \in \mathcal { P } _ { k }$ of user i’s request of type k on node v. These variables evolve as $\widehat { \mathbf { y } } _ { t } \sim P ( \cdot \mid \mathbf { y } _ { t - 1 } , \mathbf { o } _ { t } , \mathbf { u } _ { t - 1 } ^ { y } ) , \mathbf { d } _ { t } ^ { y } \sim P ( \cdot \mid \mathbf { d } _ { t - 1 } ^ { y } , \widehat { \mathbf { y } } _ { t } , \mathbf { c } _ { t } ^ { y } )$ , and $\mathbf { y } _ { t } \sim P ( \cdot \mid \hat { \mathbf { y } } _ { t } , \mathbf { d } _ { t } ^ { y } )$ , factorized over users, nodes, task types, and subtasks. The scheduler action $\mathbf { u } _ { t - 1 } ^ { y , \pi }$ assigns, cancels, or finalizes subtasks, while the executor action ${ \bf u } _ { t - 1 } ^ { y , \varsigma }$ can abort running subtasks. Assigning a subtask starts with download on user nodes and directly with execution on the server $v _ { 0 } ;$ ofline user nodes reset non-completed subtasks to idle, whereas done subtasks can still be finalized because the server already has their outputs.

The duration $d _ { t , i , v , k , p } ^ { y }$ tracks the current active phase $y \in \tilde { \mathcal { Y } } = \{ \mathtt { d o w n l o a d } .$ execution, upload . It increments while the phase continues and resets to 0 when the phase completes, is interrupted, or reaches done. The actual state $\mathbf { y } _ { t }$ copies $\hat { \mathbf { y } } _ { t } ,$ , except that duration resets advance active phases as download execution, execution  upload and upload  done on user nodes, and execution  done on the server. In the implementation, we model phase completion with discrete-time hazard models [23] fitted separately for each $y \in \tilde { \mathcal { D } }$ We fit these models from measured executions of representative computer-vision subtasks under diferent bandwidth levels, batch sizes 1, 2, 5, 10, 20, 50 , and subtask types. The covariates include standardized resource consumptions $\mathbf { c } _ { t } ^ { y } .$ batch size, log $d _ { t , i , v , k , p } ^ { y } ,$ polynomial interactions up to degree $q \in \{ 1 , 2 , 3 \}$ , and one-hot subtask-type indicators; we select a common degree q by average AIC across phases. During simulation, subtasks with the same node, subtask type, and execution age form a batch, share one sampled completion event, and either advance by resetting $d ^ { y }$ to 0 or continue with $d ^ { y } + 1$

Resource consumption $\left( \mathbf { c } _ { t } \right)$ . We define total resource consumption as $\mathbf { c } _ { t } ~ = ~ \langle \mathbf { c } _ { t } ^ { x } , \mathbf { c } _ { t } ^ { y } \rangle$ , where $\mathbf { c } _ { t } ^ { x } ~ \in ~ \mathbb { R } _ { > 0 } ^ { | \mathcal { \bar { R } } ^ { x } | \times | \mathcal { V } | \times | \mathcal { K } | \times P _ { \operatorname* { m a x } } }$ captures subtask-data preparation and $\mathbf { c } _ { t } ^ { y } \ \in \ \mathbb { R } _ { > 0 } ^ { | \mathcal { R } ^ { y } | \times | \mathcal { T } | \times | \mathcal { V } | \times | \mathcal { K } | \times P _ { \operatorname* { m a x } } }$ captures request execution. We use $\mathcal { R } ^ { x } = \mathcal { R } \backslash$ band out and $\mathcal { R } ^ { y } = \mathcal { R } \backslash$ storage , with execution-stage subsets for downloading, computation, and uploading. The variables evolve jointly as $( \mathbf { c } _ { t } ^ { x } , \mathbf { c } _ { t } ^ { y } ) \sim P ( \cdot \mid \tilde { \mathbf { c } } _ { t - 1 } ^ { x } , \tilde { \mathbf { c } } _ { t - 1 } ^ { y } , \mathbf { a } _ { t } , \hat { \mathbf { x } } _ { t } , \hat { \mathbf { y } } _ { t } , \mathbf { d } _ { t - 1 } ^ { x } , \mathbf { d } _ { t - 1 } ^ { y } )$ , where the history windows $\tilde { \mathbf { c } } _ { t - 1 } ^ { x }$ and $\tilde { \mathbf { c } } _ { t - 1 } ^ { y }$ capture autocorrelation. We condition on allocated resources because they bound feasible consumption, on auxiliary states because inactive subtasks consume no resources, and on durations because subtasks with the same phase age may share resources through batching. Server and user consumptions are coupled: user transfers consume server bandwidth, while limited server bandwidth constrains user-side transfer rates.

In the implementation, we fit generative ARMA models to measured consumption traces after resampling to 1 s, standardization, and Yeo–Johnson transformation. For $\mathbf { c } _ { t } ^ { x }$ , we collect traces by downloading subtask data in 1 Kb chunks under representative inbound bandwidth levels obtained by applying Lloyd– Max quantization [14] to the inbound-bandwidth traces collected for allocatedresource modeling. We model storage as a delta and accumulate it over time. For $\mathbf { c } _ { t } ^ { y }$ , we collect traces from representative subtasks under diferent bandwidth levels, batch sizes $\{ 1 , 2 , 5 , 1 0 , 2 0 , 5 0 \}$ , and subtask types. For each resource and stage, we fit ARMA candidates with $p \in \{ 1 , 2 , 5 , 1 0 \}$ and $q \in \{ 0 , 1 , 2 , 5 , 1 0 \}$ selecting the model with the lowest AIC. During simulation, we sample nodelevel consumption, interpolate scaling statistics over the relevant bandwidth or batch-size levels using inverse-distance weighting [22], clip transfer consumption to the efective bandwidth shared with the server, proportionally downscale CPU-dependent throughput when CPU is overcommitted, and divide node-level consumption among concurrent transfers or same-age execution batches.

## 2.3 Decision-Making Policies

The system uses two types of policies. The server-side scheduler π issues readiness and execution actions, $\mathbf { u } _ { t } ^ { x , \pi }$ and $\mathbf { u } _ { t } ^ { y , \pi }$ , while the node-side executor ς issues constraint-enforcement actions, ${ \mathbf { u } } _ { t } ^ { x , S }$ and ${ \mathbf { u } } _ { t } ^ { y , \varsigma }$ . Their action domains are $\mathcal { U } _ { x , \pi } =$ none, download, cancel , $\mathcal { U } _ { y , \pi } =$ none, execute, finish, cancel , $\mathcal { U } _ { x , \varsigma } = \{ \mathtt { n o n e }$ abort, pause, resume , and $\mathcal { U } _ { y , \varsigma } = \{ \mathtt { n o n e } , \mathtt { a b o r t } \}$ . Both policies condition on the current state excluding the actions being generated, i.e., $\mathbf { u } _ { t } ^ { \pi } \sim \pi ( \cdot \mid \mathbf { s } _ { t } \setminus \mathbf { u } _ { t } )$ and $\mathbf { u } _ { t } ^ { \varsigma } \sim \varsigma ( \cdot \mid \mathbf { s } _ { t } \setminus \mathbf { u } _ { t } )$ . Executor actions take precedence over scheduler actions.

Executor policy. The executor policy ς enforces local feasibility by comparing total consumption from $\mathbf { c } _ { t } ^ { x }$ and $\mathbf { c } _ { t } ^ { y }$ with allocated resources $\mathbf { a } _ { t }$ . Under non-storage overflow, it pauses all active subtask-data downloads on the afected node, since they share a single download channel and can later resume. Under storage overflow, it aborts downloads with the largest storage footprints until feasibility is restored. It also aborts executions whose required subtask data are no longer available or have been aborted. If overflow remains, it aborts executions in the order download execution upload; for execution, it groups subtasks by node, subtask, and execution age, and greedily aborts the heaviest normalized-load batches. Paused downloads resume once the node has no active executions and no non-storage overflow.

Centralized scheduler policy. The centralized scheduler represents fully centralized inference baseline. Since all executions run on the server $v _ { 0 } .$ , which stores all subtask data a priori, it schedules no user-node downloads. For each active request $q _ { t , i } = k$ , it cancels obsolete executions, finishes requests whose required subtasks have reached done, and assigns missing subtasks $p \in \mathcal { P } _ { k }$ only to $v _ { 0 } .$ , subject to available server memory estimated by $\widehat { m } _ { k , p , n } ^ { y }$

Uniform scheduler policy. The uniform scheduler uses online volunteered nodes opportunistically. For each online node, the policy schedules downloading at most one missing subtask data whose storage requirement and conservative download-memory estimate $\widehat { m } ^ { x }$ fit within the node’s free capacity; the server is excluded because it already stores all subtask data. The policy cancels obsolete executions, finishes completed requests, identifies requested but unplaced user– subtask pairs, and assigns them uniformly at random among feasible nodes. A node is feasible if it is online, has $x _ { t , v , k , p } = \mathtt { r e a d y } _ { \mathrm { : } }$ , and has enough free memory for the marginal execution-memory estimate $\widehat { m } _ { k , p , n } ^ { y }$ . We precompute $\widehat { m } ^ { x }$ and $\widehat { m } _ { k , p , n } ^ { y }$ from learned memory models using a $\mu + 2 \sigma$ safety margin, interpolating $\widehat { m } _ { k , p , n } ^ { y }$ across measured batch sizes. Memory estimates are needed because memory, unlike CPU and bandwidth, is a hard constraint.

Afinity-based scheduler policy. The afinity-based scheduler uses the same download, cancellation, and finishing logic as the uniform policy, but assigns executions to improve batching. For each required subtask $( k , p )$ , it prioritizes eligible nodes by the number of active executions of the same subtask already placed on them, excluding ofline nodes and nodes without ready subtask data. Using $\widehat { m } _ { k , p , n } ^ { y } ,$ it computes how many additional executions fit in each eligible node’s free memory and fills the resulting assignment slots with randomly shufled users requiring $( k , p )$ . This greedily co-locates same-subtask executions to increase batch sizes while aiming to preserve memory feasibility.

## 2.4 Optimization Problems

Given the proposed system model, two optimization problems naturally arise. First, for a fixed executor policy $\varsigma ,$ one can optimize the scheduler policy π to maximize QoS, e.g., by minimizing long-term average request latency:

$$
\operatorname* { m a x } _ { \pi } \operatorname* { l i m i n f } _ { T  \infty } \mathbb { E } [ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } r ( \mathbf { s } _ { t } ) \mid \mathbf { s } _ { 0 } , \mathbf { u } _ { 0 : t - 1 } ^ { \pi } \sim \pi ] ,\tag{2}
$$

where $r ( \mathbf { s } _ { t } ) \triangleq - \mathbf { 1 } ^ { \top } \delta ( \mathbf { q } _ { t } , \mathbf { d } _ { t } ^ { q } )$ is a reward function, and $\delta ( \mathbf { q } _ { t } , \mathbf { d } _ { t } ^ { q } ) \in \{ 0 , 1 \} ^ { | \mathcal { T } | }$ denotes per-user latency increments.

Second, for a fixed scheduler π and executor policy ς, one can optimize the allocation of dedicated server resources. In this case, the objective is to determine the amount of dedicated capacity required to satisfy a target QoS constraint under the given model and policies; e.g., one may require request latency to remain below 10 s for 99% of requests, corresponding to P99 latency. We define this problem as

$$
\begin{array} { l } { \underset { \mathbf { a _ { f } } } { \mathop { \operatorname* { m i n } } } \left\| \mathbf { a _ { f } } \right\| _ { \mathbf { w } } } \\ { \mathrm { s . t . } \quad \mathbb { P } \big ( d _ { t , i } ^ { q } \leq \tau _ { k } \mid q _ { t + 1 , i } = k _ { 0 } , q _ { t , i } = k \big ) \geq 1 - \epsilon _ { k } , \quad \forall k \in \mathcal { K } , } \end{array}\tag{3}
$$

where $\mathbf { a } _ { \dagger }$ is a nominal, $\mathrm { i . e . , }$ maximum, server resource capacity that parametrizes transition probabilities of $\mathbf { a } _ { t } ,$ and $\| \cdot \| _ { \mathbf { w } }$ is a weighted $\ell ^ { 1 }$ norm with per-resource cost weights $\mathbf { w } , \tau _ { k }$ are per-request-type latency thresholds, and $\epsilon _ { k } \in ( 0 , 1 )$ are significance levels. Formal definitions of these problems are provided in [12].

It is noteworthy to mention that this paper does not focus on designing a scheduling algorithm or resource-allocation optimizer. Instead, our primary focus is to define the collaborative distributed inference system and develop a model of its dynamics. We use this model to simulate fixed policies under diferent configurations and compare collaborative distributed inference with centralized inference. Future work will use this model to address the optimization problems.

## 3 Evaluation

## 3.1 Experimental Setup

We evaluate the proposed system by simulating trajectories of the generative model, sampling $\mathbf { s } _ { t } \sim P ( \cdot \mid \mathbf { s } _ { t - 1 } )$ . We parameterize ARMA resource-consumption and hazard-based duration models from measurements collected on a MacBook M4 Pro with 24 GB RAM and 512 GB storage. We consider four task types: light and heavy, each with 5 or 10 subtasks, sampled with probabilities 0.4, 0.3, 0.2, 0.1 to reflect that simpler tasks are typically requested more often. Subtasks are instantiated using established computer-vision models. Each simulation runs for 100,000 time steps, or approximately 28 hours. We vary user count 100, 1,000, 5,000, 10,000 , user capacity $\{ 1 \times , 2 \times \}$ of base resource profile, server capacity $\{ 1 0 \times , 3 0 \times , 5 0 \times , 8 0 \times , 1 0 0 \times \}$ , and scheduler policy. These dimensions test how the system behaves as demand and volunteered resources grow together, how ofloading depends on user-device capacity, how much dedicated capacity is needed to sustain QoS, and how scheduling afects system behavior.

We define the base resource profile as a moderate, non-intrusive contribution from a user device: 100 Mbit inbound/outbound bandwidth, 3 GB storage, 2.5 GB memory, and 200% CPU. The bandwidth represents neither a weak nor exceptional connection; the storage can hold several small subtasks or one large subtask with a few smaller ones. We set memory and CPU to rounded values slightly above the largest measured average subtask consumption at batch size 2, making small batches of demanding subtasks feasible with high probability and larger lightweight batches more likely. These values define the maximum allocation thresholds $\overline { { a } } _ { r , v } ^ { o } ,$ while realized capacities $\mathbf { a } _ { t }$ still vary over time. We report request completions and cancellations, P99 latency as the primary QoS metric, dedicated resource consumption, and server/user subtask completion rates. We focus on P99 latency because other latency summaries, such as mean and P95 latency, showed similar trends and are omitted due to space constraints.

![](images/caab264bf7e4081a321f3cd7b5286e76ead227e100bf7b56cdded468da2ccd8f.jpg)  
(a) Server 10×

![](images/b11d1ac173841ea39a18ecd9014673e061dd0eb84f032797219dcbc84d060ab6.jpg)  
(b) Server 30×

![](images/aebf327330af1b3cb2e283c06b479c889671e1b42f68526edad8c304a5a84c76.jpg)  
(c) Server 50×

![](images/cd1e07f6c6554fc676980c5920fcab3724c4a7e8493aa41fdeecf7d3a921c59c.jpg)  
(d) Server 80×

![](images/75a4ae61358a0f98a06494c2c24b75ab362d5fae1b701187e8c9e3a8e7404c74.jpg)  
(e) Server 100×  
Fig. 2: Completed vs. canceled requests under diferent scheduler policies and server capacities. Each subfigure corresponds to a diferent server capacity multiplier. The x-axis varies the user configuration, i.e., the number of users ( ) and the per-user resource-capacity multiplier.

For full reproducibility, we published our implementation code on GitHub.<sup>1</sup>

## 3.2 Results

Completed vs. canceled tasks. Fig. 2 compares completed and canceled requests. The centralized policy is unafected by user capacity because all subtasks run on the server. Under the centralized policy, at 10 capacity, completions grow only through 1,000 users, then plateau and decline as cancellations surge. Higher capacities delay saturation, but gains diminish beyond 30 ; at large user counts, cancellations still equal roughly one third to one half of completions.

The uniform policy matches the centralized through 1,000 users and improves with 2 user capacity. At 5,000 users, it performs better at 10 and 30 server capacity and comparably above them. At 10,000, it yields more completions and fewer cancellations at every server capacity, including 100 . Thus, the growing volunteered resource pool ofsets demand more efectively than centralized scaling, whose gains from 50 to 100 are incremental. Centralized execution may also form larger batches that save resources but delay individual subtasks, whereas uniform distribution creates smaller batches across more nodes.

The afinity-based policy is competitive through 1,000 users and beats the centralized policy at 10 capacity, but otherwise generally yields the worst completion-to-cancellation balance. Its larger batches may extend execution and increase cancellation risk; node departures or executor aborts can also interrupt several batched subtasks simultaneously. Overall, distributed inference is efective and feasible, but batching eficiency must be balanced against latency and failure risk, motivating scheduler optimization using the proposed model.

![](images/5941417d058ed3efc8cc903b76b32b036d2b4a34cefecb48105d9524245ade19.jpg)

![](images/de6692b490ec55b92d2ca9bc744126ed596a698a4c2830d63b87a225f6cb50e6.jpg)  
(a) Server 10×  
(b) Server 30×

![](images/9a740bf257824f15c81620cb2afff92e1146fe1f429ee8e2ce09c932c4db0994.jpg)  
(c) Server 50×

![](images/c4b81314d216071f0d6ba19f4ac453cc74ae42a361edcecc3fa170ac9bcc314e.jpg)  
(d) Server 80×

![](images/2f74dbf014201cc88cb0a5520c7d889cadab6c0d30f9c7d6b292b3fdb6a6c90e.jpg)  
(e) Server 100×  
Fig. 3: P99 request latency under diferent scheduler policies and server capacities. Each subfigure corresponds to a diferent server capacity. The x-axis shows user configurations. The curves represent average P99 latencies across request types. The faded regions represent the range of P99 latencies across request types, from the lowest to the highest.

Task latencies. Fig. 3 reports P99 request latency. The centralized policy performs best or comparably through 1,000 users, and at 5,000 as server capacity grows. At 10,000, the uniform policy remains substantially better. Centralized latency varies more across request types under higher demand. Besides diferences in task complexity, unequal request probabilities lead to frequent subtasks forming larger batches that run longer and reduce the resources available to other requests. More server capacity mitigates but does not eliminate this efect.

The uniform policy matches the centralized policy through 1,000 users, outperforms it at 5,000 with up to 30 server capacity, and remains comparable above that. At 10,000, it consistently achieves much lower P99 latency. Its variation under constrained configurations likely arises because weaker users store fewer models, particularly for heavy tasks, forcing more execution onto the server. Increasing user and server capacities distributes more subtasks and largely removes this variation.

At 10 server capacity, the afinity-based policy is slightly worse than the uniform policy but better than the centralized policy at large user counts. Beyond 30 , its latency changes little and generally remains above the uniform policy, likely because larger batches delay and couple multiple requests. Overall, both distributed policies show little improvement beyond 30 –50 server capacity, i.e., this range provides suficient dedicated capacity to complement volunteered resources. Centralized scaling improves slowly and does not surpass the uniform policy at a large scale. Thus, distributed scheduling provides better tail latency and under lower server capacity than centralized scheduling.

Dedicated resource consumption. Fig. 4 shows that the centralized policy consistently uses more dedicated resources. CPU consumption reaches every allocation limit, indicating persistent saturation. Since consumption is recorded before the executor resolves hard-constraint violations, mean memory demand reaches nearly twice the allocated 25 GB at 10 capacity and slightly exceeds 75 GB at 30 , causing the subtask aborts. At higher capacities, memory remains within its allocation and grows slowly, suggesting that demand approaches its required level while additional CPU accelerates processing, causing slight further increases in memory usage.

![](images/d7031bc70eb46deda61766d38c0262e14d71d1c6d4ac57d2df6e91e39823b018.jpg)  
(a) Server 10×

![](images/3c11f8850dc5cf5a46451ce80a526111ca5c4f937efcf5cdd74fe48678d8b21d.jpg)  
(b) Server 30×

![](images/0aafe794eeb3a680196ca2ce86128ce680c251ee551e8bd44160841c53535a2a.jpg)  
(c) Server 50×

![](images/ab45af4f21eed7de26a9087075c5e21d2d6d02d430699330ef7600546647f8be.jpg)  
(d) Server 80×

![](images/c2001eb145b5ee06d3d049d6d260d18564f3c958668f13521891e8d625f71c4b.jpg)  
(e) Server 100×

![](images/b593fef8f9c2af86017048bf0bf5c47c7bf1142d8d5fb9824b442f64cc51e93b.jpg)  
(f) Server 10×

![](images/8548bdb715c5afb669f2803d95b827172d010d272edc9c992f4d3ed4f7766897.jpg)  
(g) Server 30×

![](images/e2ab38d6e6117dcd40a55518335e9c70af0360b0589660296acccbd654e05649.jpg)  
(h) Server 50×

![](images/ec901b72d8e968ea39c0a4dc1767b7dd959ef0a394c43c17aadf545292bc5795.jpg)  
(i) Server 80×

![](images/960f58e87d1397e26ccc0c2c1751a8ab8e0707643f5998b848b92e324e8a9519.jpg)  
(j) Server 100×  
Fig. 4: Server resource consumption under diferent scheduler policies and server capacities. The first row reports CPU utilization, where 100% corresponds to full utilization of one CPU core, and the second row reports memory usage. The curves represent average resource consumptions over time. The faded regions represent the corresponding standard deviations.

Both distributed policies use substantially fewer dedicated resources and never exceed their allocations. Consumption changes little between 30 and 100 , indicating that approximately 30 –50 capacity suficiently complements volunteered resources. The afinity-based policy is slightly more eficient than the uniform policy, likely because it considers node loads and exploits batching. Overall, distributed scheduling accommodates growing demand without proportional expansion of dedicated infrastructure.

Subtask completion. Fig. 5 reports subtask completion rates on server and user nodes; the centralized policy is omitted from user-node plots. Under distributed policies, user-side rates are largely independent of server capacity. The uniform policy achieves approximately 0.75–0.95, improving with user capacity because stronger nodes better tolerate resource fluctuations. Afinity-based scheduling performs better at 100 and 1,000 users with 1 capacity, but degrades as user count grows, especially on weaker nodes. Larger batches may exceed expected resource demands, leading to the simultaneous abort of multiple subtasks due to node departures or resource-constraint violations.

Centralized server-side rates at 10 capacity fall from 0.9–1.0 for up to 1,000 users to approximately 0.2 at 10,000. At 30 , they remain near 1.0 through 1,000 users but decline to approximately 0.6 at 10,000; only 100 capacity approaches consistently complete execution. Distributed policies reach near-perfect serverside rates with considerably less capacity, particularly for 2 users, because volunteered resources reduce server overcommitment. Some redundant work remains because rule-based schedulers cannot anticipate availability and resource fluctuations. Overall, distribution reduces latency and dedicated capacity needs, showing the value of scheduler optimization.

![](images/f404749fa664048c0aece806e3ba7f1bfb2bb46756f18547061ee8bfb17a06db.jpg)

![](images/d85b88af55fc748c20998b4c33d201e848626055c3c3d4e83aec1779be387f31.jpg)  
(b) Server 30×

![](images/03e98e0ccb0cba9de4c6f595b8e71b1af69bf537f9dcac1aefc29e7cca90e5dd.jpg)  
(a) Server 10×  
(c) Server 50×

![](images/bf740823cac431181e2e1dc9f52aaa2cb78eb92217993c5ddba6133517ddd48a.jpg)  
(d) Server 80×

![](images/ca99964721b3d2f0fc121d5a71d5a85092db344f7373f4b032d19440c2a55df9.jpg)  
(e) Server 100×

![](images/04ccf82512e38bd81c53bcab9eabfd815cd61ae66e418b96d8b35ea9ef32ad52.jpg)  
(f) Server 10×

![](images/ffadce7c331529408d7fad335dbf854bdece7b2417858ddcd7fb3eeb8c39a4f2.jpg)  
(g) Server 30×

![](images/f3dead5ee853f1b7b335a86a03a948d646bb07d21eb61479884634be67ea0cae.jpg)  
(h) Server 50×

![](images/90268658b1420ea6cc63c9507d4e04f252d404e8b61bc3ef964e4e9e41c33fa1.jpg)  
(i) Server 80×

![](images/f6a633f1e8ce34d2249af35bb4a8ca99850cc60798f104ad9cf60a33d5702bde.jpg)  
(j) Server 100×

Fig. 5: Subtask completion rates under diferent scheduler policies and server capacities. Completion rate denotes the fraction of scheduled subtasks completed successfully. The first and second rows show completion rates on the server and user nodes, respectively.

## 3.3 Limitations and Future Work

This paper focuses on the collaborative distributed inference architecture and its modeling framework, instead of a production-ready implementation. Accordingly, we assume homogeneous users with identical availability and resource distributions, which also allows us to omit explicit fairness mechanisms. A real deployment could condition user behavior and resources on temporal and userspecific context and incorporate contribution-aware request prioritization. We further use 1 s time steps from the server’s perspective; irregular observations can be aligned with this representation through interpolation.

Although the conceptual execution state grows quadratically with the user count, our implementation uses limited slots to bound the tasks maintained per node and scale linearly. Alternative request-centric representations could further reduce this space cost, making model scalability and decentralization promising research directions. The framework can also naturally extend to multiple dedicated servers, DAG-based workflows, and autoregressive workloads such as LLM inference. Finally, optimizing the scheduler and dedicated resource allocation formulated in Section 2.4, as well as addressing deployment-level security and privacy, remains future work. These promising extensions build directly on the proposed model, whose modular factorization allows individual assumptions and components to be refined without redesigning the overall framework.

## 4 Conclusion

We proposed a collaborative distributed inference system that combines dedicated server infrastructure with resources contributed by users’ devices, enabling demand to scale without proportional growth in dedicated capacity. To capture the system’s stochastic and dynamic behavior, we developed a generative

Markov model that supports simulation, analysis, and policy optimization. Our evaluation showed that distributed scheduling increasingly outperforms centralized inference as the user population grows, improving request completion and tail latency while substantially reducing dedicated resource consumption. These findings establish the feasibility of user-assisted inference and provide a foundation for future work on optimized scheduling, QoS-aware resource allocation, and extensions to richer workloads and deployment environments.

Acknowledgments. The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725 and also supported by MSCA’s TALENTS (101299722) project.

## References

1. Anderson, D., Burnham, K.: Model selection and multi-model inference. Second. NY: Springer-Verlag 63(2020), 10 (2004)

2. Benevenuto, F., et al.: Characterizing user behavior in online social networks. In: Proceedings of the 9th ACM SIGCOMM Conference on Internet Measurement. p. 49–62. IMC ’09, ACM, New York, NY, USA (2009). https://doi.org/10.1145/ 1644893.1644900

3. Box, G.E., et al.: Time series analysis: forecasting and control. John Wiley & Sons (2015)

4. Che, D., Hou, W.C.: A novel “credit union“ model of cloud computing. In: Cherifi, H., Zain, J.M., El-Qawasmeh, E. (eds.) Digital Information and Communication Technology and Its Applications. pp. 714–727. Springer, Berlin, Heidelberg (2011)

5. Crovella, M., Bestavros, A.: Self-similarity in world wide web trafic: evidence and possible causes. IEEE/ACM Transactions on Networking 5(6), 835–846 (1997). https://doi.org/10.1109/90.650143

6. Ding, A.Y., et al.: Roadmap for edge ai: a dagstuhl perspective 52(1), 28–33 (Mar 2022). https://doi.org/10.1145/3523230.3523235

7. Donta, P.K., et al.: Exploring the potential of distributed computing continuum systems. Computers 12(10) (2023). https://doi.org/10.3390/ computers12100198

8. Dustdar, S., et al.: On distributed computing continuum systems. IEEE Transactions on Knowledge and Data Engineering 35(4), 4092–4105 (2023). https: //doi.org/10.1109/TKDE.2022.3142856

9. Gao, Y., et al.: Optimizing multi-dnn inference on mobile devices through heterogeneous processor co-execution. IEEE Transactions on Mobile Computing 25(6), 7766–7781 (2026)

10. Javadi, B., et al.: Discovering statistical models of availability in large distributed systems: An empirical study of seti@home. IEEE Transactions on Parallel and Distributed Systems 22(11), 1896–1903 (2011). https://doi.org/10.1109/TPDS. 2011.50

11. Jin, Y., Yang, Z.: Scalability optimization in cloud-based ai inference services: Strategies for real-time load balancing and automated scaling. In: Proceedings of the 2025 4th International Conference on Big Data, Information and Computer Network. p. 266–270. BDICN ’25, ACM, New York, NY, USA (2025). https: //doi.org/10.1145/3727353.3727398

12. Lapkovskis, A., et al.: Brief announcement: Generative markov model for distributed computing systems. arXiv preprint arXiv:2606.03061 (2026)

13. Li, P., et al.: Making ai less ’thirsty’. Commun. ACM 68(7), 54–61 (Jun 2025). https://doi.org/10.1145/3724499

14. Lloyd, S.: Least squares quantization in pcm. IEEE Transactions on Information Theory 28(2), 129–137 (1982). https://doi.org/10.1109/TIT.1982.1056489

15. Ma, P., et al.: Research allocation in mobile volunteer computing system: Taxonomy, challenges and future work. Future Generation Computer Systems 154, 251– 265 (2024). https://doi.org/https://doi.org/10.1016/j.future.2024.01.015

16. MARZBAN, M.F.A., et al.: Inference data distribution criteria for artificial intelligence or machine learning model monitoring (Jun 19 2025), uS Patent App. 18/544,782

17. Mell, P.M., Grance, T.: Sp 800-145. the nist definition of cloud computing. Tech. rep., National Institute of Standards & Technology, Gaithersburg, MD, USA (2011)

18. Mengistu, T.M., Che, D.: Survey and taxonomy of volunteer computing. ACM Comput. Surv. 52(3) (Jul 2019). https://doi.org/10.1145/3320073

19. Mengistu, T.M., et al.: cucloud: Volunteer computing as a service (vcaas) system. In: Luo, M., Zhang, L.J. (eds.) Cloud Computing – CLOUD 2018. pp. 251–264. Springer International Publishing, Cham (2018)

20. Meuser, T., et al.: Revisiting edge ai: Opportunities and challenges. IEEE Internet Computing 28(4), 49–59 (2024). https://doi.org/10.1109/MIC.2024.3383758

21. Sajadieh, S., et al.: The AI index 2026 annual report. Tech. rep., AI Index Steering Committee, Institute for Human-Centered AI, Stanford University, Stanford, CA (April 2026)

22. Shawky, M.M.: A comparative study of interpolation methods for the development of ore distribution maps. Discover Geoscience 3(1), 2 (2025)

23. Singer, J.D., Willett, J.B.: Applied longitudinal data analysis: Modeling change and event occurrence. Oxford university press (2003)

24. Wang, W., et al.: A survey of ai inference technologies for on-device systems. IEEE Internet of Things Journal 12(24), 51927–51950 (2025). https://doi.org/ 10.1109/JIOT.2025.3613691

25. Yeo, I., Johnson, R.A.: A new family of power transformations to improve normality or symmetry. Biometrika 87(4), 954–959 (12 2000)

26. Zhang, G., et al.: Task-aware collaborative inference and fine-grained dnn partitioning in mec networks. IEEE Transactions on Mobile Computing 25(6), 8911–8927 (2026). https://doi.org/10.1109/TMC.2025.3650680

27. Zhang, T., et al.: Dishelis: Optimizing deployment of disaggregated llms inference serving over heterogeneous environments via hierarchical max-flow. IEEE Transactions on Cognitive Communications and Networking 12, 5473–5488 (2026). https://doi.org/10.1109/TCCN.2026.3657037

28. Zheng, T., et al.: Joint optimization of dynamic batching and adaptive partitioning for distributed llms inference in mobile edge computing. IEEE Transactions on Mobile Computing 25(6), 8747–8763 (2026). https://doi.org/10.1109/TMC. 2026.3650838