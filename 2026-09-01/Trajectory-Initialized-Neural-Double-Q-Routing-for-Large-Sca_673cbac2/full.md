# Trajectory-Initialized Neural Double Q-Routing for Large-Scale Overhead Hoist Transport Systems

Cheng Gu, Qiusheng Zhao, Anbang Liu, Shaochong Lin, and Max Z. J. Shen

Abstract—Large-scale industrial robot fleets share constrained physical infrastructure, making vehicle travel times dependent on safety separation, intersection access, downstream blocking, and station contention. We study this problem in overhead hoist transport (OHT) systems, a representative ceiling-mounted material-handling system used in semiconductor fabs. Static shortest-path routing cannot account for these time-varying traffic costs, whereas tabular Q-routing adapts online but learns each destination–node–action value independently, limiting information sharing across sparsely visited routing contexts and making startup behavior sensitive to inaccurate value estimates. We propose Neural Double Q-routing, which replaces destinationindexed tables with a shared state–action value network. The network is warm-started through return-to-go regression on mixed simulator-generated routing trajectories and then refined online using Double-Q updates, local congestion correction, and event-stratified structured replay. Across nine matched fleet-size– arrival-rate settings with 100, 150, and 200 OHTs, the proposed framework reduces mean completion time relative to tabular Double Q-routing by 0.8%–8.8%. It achieves the lowest mean completion time among all compared methods in the six 150- and 200-OHT settings, whereas Dijkstra remains best in the three 100-OHT settings. Completed-task counts remain within 1% of tabular Double Q-routing in eight of nine settings, and 95th-percentile completion time decreases in eight settings. In two matched startup scenarios, offline initialization increases the number of completed tasks by up to 23% and reduces tail completion time by up to 15%.

Note to Practitioners— Overhead hoist transport controllers in semiconductor factories must select routes quickly while travel conditions change because of vehicle separation, intersection access, downstream queues, blocking, and load/unload contention. Conventional shortest-path routing is straightforward to deploy but does not respond to these delays, whereas table-based adaptive routing may require substantial online experience before its route estimates become dependable. The proposed controller changes only the nexthop decision at track split nodes; existing low-level mechanisms for vehicle separation, intersection access, and other safety functions remain in place. A compact neural value model is initialized from simulator-generated routing trajectories and subsequently updated during operation. In the evaluated simulations, its benefit depended on the operating condition: it achieved the lowest mean completion time in the 150- and 200-OHT settings, while Dijkstra remained best in the 100-OHT settings. Offline initialization also increased startup task completions and reduced tail completion time in the two evaluated startup scenarios. Deployment requires a representative simulator, reliable local traffic measurements, and facility-specific validation

of reward settings and controller interfaces. The present evidence is limited to one fab layout and one simulator; physical commissioning, initialization from operating logs, transfer to other transport systems, and robustness to layout changes remain to be evaluated.

Index Terms—Automated material handling systems, dynamic routing, overhead hoist transport, reinforcement learning, semiconductor manufacturing.

## I. INTRODUCTION

In a 300 mm semiconductor fab, an automated material handling system moves front-opening unified pods among process tools, stockers, and buffers to sustain continuous production. Overhead hoist transport (OHT) vehicles execute these moves on a ceiling-mounted, unidirectional guideway (Fig. 1). Once a transport task and its current pickup or delivery destination have been assigned, the routing controller repeatedly selects an outgoing track at each split. These choices determine when the carrier reaches its endpoint and how long the vehicle occupies shared guideway capacity. OHT routing is therefore a repeated resource-allocation decision within the fab transport schedule, while task dispatching remains separate.

The central difficulty is endogenous congestion. Safety spacing, exclusive intersection access, shared track segments, downstream admission constraints, and finite load/unload capacity couple vehicle motion [1]. Consequently, the elapsed time after a route choice contains free-flow motion, waiting for separation or right-of-way, downstream blocking, queue spillback, and station contention [2], [3]. A geometrically short route can direct a vehicle into a queue and propagate delay to following vehicles, whereas a locally longer route can avoid a contested segment. Next-hop consequences are therefore traffic-dependent and fleet-coupled. Yet each vehicle must choose from graph-feasible outgoing edges using bounded current information, while the existing low-level controller continues to enforce vehicle separation and intersection access.

A joint look-ahead controller would have to anticipate future task releases, interacting vehicle trajectories, and timedependent resource availability across a large network. Practical OHT routing therefore commonly relies on computationally light rules, shortest paths, or dynamically adjusted edge costs [4], [5]. Static shortest-path routing is deterministic and easy to audit, but fixed costs cannot represent current traffic. Congestion-aware routing responds to measured or predicted conditions, yet it compresses those conditions into prescribed edge penalties whose quality determines how well downstream route-choice effects are represented. We assess routing primarily by mean task completion time, while completed-task count and 95th-percentile completion time reveal unfinished work or delay shifted into the tail.

![](images/99b1956ef903477a65831134ab1d7e59f9cc29dac2173b9c8b146b4697ccf38d.jpg)

![](images/8d775a791bad930998cd3d73996cb358dff4a42fa813204e5d7b54968895c6be.jpg)  
Fig. 1. OHT routing context and decision process. (a) Vehicles move carriers along a ceiling-mounted, unidirectional guideway where safety spacing, shared segments, and exclusive intersections couple travel times across the fleet. (b) On the directed guideway graph, an OHT with destination D makes a local next-hop decision at split node i, choosing between candidates such as j<sub>1</sub> and j<sub>2</sub>, and repeats these decisions to form a complete route from start to destination.

Q-routing addresses the downstream-effect problem by learning the expected remaining travel time after a vehicle at node i selects neighbor j toward destination d [2], [6]. Its destination-conditioned table supports fast local decisions and adapts from realized transitions. However, every destination– node–neighbor tuple $( d , i , j )$ has an independent value, so one update cannot directly inform another tuple with similar route progress and local traffic. On the evaluated guideway, 608 destinations and 4257 directed edges require approximately 2.59 million tabular Q entries, and tabular Double Q-routing doubles that storage. Frequently visited tuples may stabilize while rarely visited tuples remain data-poor. A shared candidateedge scorer instead updates one function used across nodes and destinations, although parameter sharing alone does not establish value accuracy. We retain Q-routing’s local feasibleaction interface but learn discounted return under a composite routing reward rather than remaining travel time alone.

Replacing the table with a neural function addresses only the representation problem. A continuing online router must also be useful before accumulating extensive new experience and must adapt without becoming unstable or unresponsive.

![](images/a5a7718f812dd0b282101d6a2e0d04ca14b4bb5c96527dd7eb43f8aa133b6e10.jpg)  
Fig. 2. Tabular versus neural Q-routing. Tabular Q-routing stores an independent value for each (d, i, j) tuple, whereas a neural value function Q (s, a) scores candidate actions from a common state–action feature vector ϕ(s, a). Shared parameters let observations from frequently visited contexts inform the fitted scoring rule used at less frequently visited contexts.

Random initialization provides no informative startup ranking, and using one noisy estimator for action selection and evaluation can introduce positive bootstrap bias [7]. A fleet-wide value model can also lag behind short-lived local queues, while uniform replay may underrepresent infrequent blocking, neardeadlock, and load/unload bottlenecks. The design problem is therefore to combine a compact shared representation with informative initialization and stable yet responsive online refinement.

We propose Neural Double Q-routing, an offline-warmstarted, online-adaptive next-hop router for fab-scale OHT systems. A shared neural network scores each controller-admitted edge from task, topology, route-progress, and bounded localtraffic features, while transitions from all vehicles update the same value function. Before operation, supervised returnto-go regression on simulator-generated Dijkstra, Q-routing, and Double Q-routing trajectories supplies an initial estimate. Online, Double-Q targets separate action selection from evaluation, recent edge-level residuals correct local rankings without entering the recursive bootstrap, and event-stratified structured replay retains transitions from predefined congestion and service-event classes. The model changes only splitnode route selection; low-level safety controls remain outside the value function.

We evaluate the framework in one discrete-event simulator on a fab guideway with 3684 nodes, 4257 directed edges, and 608 destinations. Nine matched settings combine 100, 150, and 200 OHTs with arrival rates of 1.0, 1.5, and 2.0 task s<sup>−1</sup>. Relative to tabular Double Q-routing, Neural Double Q-routing lowers mean completion time in all nine settings by 0.8%–8.8%; completed-task counts remain within 1% in eight, and 95th-percentile completion time decreases in eight. It obtains the lowest mean among all methods in the six 150- and 200-OHT settings, whereas Dijkstra remains best in the three 100-OHT settings. The evidence therefore indicates a setting-dependent benefit rather than universal superiority. In matched startup scenes, offline initialization changes early mean completion time modestly but increases completed tasks by 22.2%–23.1% and reduces tail completion time by 7.2%– 15.0%.

The contributions are as follows:

1) A compact shared value representation for repeated OHT routing. A candidate-edge function replaces independent destination–node–neighbor entries while retaining the local graph-feasible action interface. Its parameter count is independent of |D|, |V |, and |E|, reducing 2.59 million tabular Q entries to 5825 neural parameters in the evaluated system.

2) An offline-to-online procedure for startup, stable refinement, and local responsiveness. Mixed trajectories initialize the shared value function; Double-Q targets refine global estimates; local residuals adjust recent edge rankings; and event-stratified replay retains infrequent congestion evidence.

3) A matched-scene evaluation of gains and settingdependent behavior. A fleet-size and arrival-rate sweep compares static and tabular alternatives, while warmstart and component studies separate startup effects from the online-refinement mechanisms.

The remainder of this paper is organized as follows. Section II reviews OHT control layers, congestion-aware and learning-based routing, deep traffic representations, and offline-to-online value learning. Section III formulates the event-driven OHT routing problem, including the guideway model, state–action features, decision-interval dynamics, reward, and evaluation criteria. Section IV presents the shared neural value function, offline warm start, online Double-Q refinement, local congestion correction, and structured replay. Section V describes the matched-scene protocol and reports the operating-regime sweep and ablation studies. Section VI interprets the findings and states their limitations, and Section VII concludes the paper. Appendix A provides the complexity analysis, full feature definitions, value-estimate stability analysis, and implementation details.

## II. RELATED WORK

This section positions Neural Double Q-routing along five related lines of research. We first distinguish route guidance from the neighboring OHT decisions of system design, dispatching, and rebalancing. We then review congestion-aware path construction and learning-based traffic prediction. Next, we trace the development from tabular Q-routing to neural, destination-conditioned value approximation. We subsequently discuss trajectory-based initialization, offline-to-online learning, and nonuniform experience replay. Finally, we locate the proposed method with respect to these lines in terms of its decision variable, representation, data regime, and online adaptation mechanism.

## A. Routing Within OHT Control

Manufacturing transport control spans decisions made at different temporal and spatial scales. At the production-system level, integrated scheduling methods may jointly select operations, machines, and transport resources. For example, Zhang et al. [8] use heterogeneous graph representations and deep reinforcement learning for flexible-job-shop scheduling with insufficient vehicles. Such formulations coordinate production and transport decisions rather than isolating vehicle routing.

OHT control in a semiconductor fab can itself be separated into system design, vehicle dispatching, idle-capacity rebalancing, and route guidance. Layout and simulation studies evaluate guideway capacity, traffic mechanisms, and systemlevel performance under alternative designs [1], [9]. Dispatching assigns an outstanding transport request to a vehicle. Hungarian assignment, prediction-based dispatching, and vehicle look-ahead formulations use current or anticipated requests to reduce pickup and delivery delay [3], [10], [11]. Rebalancing instead relocates idle vehicles toward areas with anticipated demand. Graph-based multi-agent policies have been developed for zone rebalancing [12] and coordinated baylevel vehicle allocation [13].

These decisions interact because dispatching and rebalancing alter the traffic encountered by the routing controller. Nevertheless, their immediate control variables differ. Dispatching determines which vehicle serves a request, rebalancing determines where idle capacity waits, and route guidance determines how a vehicle with an assigned task proceeds through the guideway. The present work fixes the task-dispatching rule and studies the repeated selection of a controller-admitted outgoing edge after a vehicle has a current task and destination.

B. Congestion-Aware Path Construction and Traffic Prediction

A major family of dynamic OHT routing methods retains a conventional shortest-path procedure while replacing static edge lengths with traffic-dependent costs. Bartlett et al. [4] update a computationally light congestion measure and reroute vehicles as measured conditions change. Gupta et al. [5] estimate congestion-dependent edge-traversal costs for Dijkstra routing and study routing costs together with pickup rules. Machine-learning traffic control has likewise been used to predict congestion at critical bottlenecks and redirect vehicles through modified routing costs [14]. In these approaches, measured or predicted traffic is translated into edge weights before a path-construction algorithm is applied.

More recent methods learn richer travel-time or traffic representations. Ahn et al. [15] represent successive OHT traffic states as graphs and use a graph-convolutional gated recurrent model to predict future edge-travel times for congestionaware rerouting. Choi et al. [16] combine a local heuristic approximation for the near portion of a candidate route with a deep neural approximation of its more distant travel time. Separable contextual graph neural networks identify spatiotemporal tailgating-oriented congestion patterns but do not themselves produce routing actions [17]. HarmonyRouting instead uses a graph neural network to estimate route-level traffic impact and incorporates that prediction into systemlevel route selection [18]. These methods exploit graph-wide or route-wide information to predict quantities subsequently used by a routing procedure.

Reinforcement learning has also been used to influence routes through intermediate traffic-control variables. Kang et al. [19] identify heavily used and frequently congested regions and use deep reinforcement learning to adjust the routing cost imposed on those regions. Koo et al. [20] partition a fab-scale guideway into areas and learn dynamic link-weight adjustments that influence downstream path selection. At a broader system level, Lee et al. [21] combine active Qrouting, discrete-event simulation, and real-time digital-twin information in an OHT orchestration framework.

The common feature of these approaches is that learning supplies a traffic prediction, path estimate, or network-level cost intervention. The routing algorithm then constructs or modifies a path using that intermediate quantity. Neural Double Q-routing exposes a different control interface: at each split, it directly evaluates the controller-admitted outgoing edges and selects one next hop from the current task, routeprogress, topology, and bounded local-traffic context.

## C. From Tabular Q-Routing to Neural Route Guidance

Q-routing was introduced for adaptive packet forwarding in dynamically changing communication networks. It stores a destination-conditioned estimate $Q ( d , i , j )$ of the remaining delivery time when an item at node i first moves to neighbor j on its way to destination d [6]. The value is updated from realized local delay and a downstream estimate, yielding an inexpensive online next-hop lookup. The original study also considered neural approximation of the routing value but reported inconclusive results, so replacing the table with a function approximator was already recognized as a possible extension. Predictive Q-routing subsequently retained favorable past estimates and modeled traffic recovery to improve adaptation when network load changes [22].

Q-routing has been adapted to semiconductor OHT systems because its local decision epoch and low online computational cost fit guideway control. Hwang and Jang [2] develop a Q(λ)-based dynamic route-guidance method with eligibility traces and exploration mechanisms. Hong et al. [23] further study Q-learning-based route guidance together with vehicle assignment in practical-scale OHT settings. These methods learn from observed OHT traffic while retaining a distinct value for each destination–node–neighbor tuple. Consequently, experience obtained at one tuple does not directly update another tuple with similar route progress or congestion conditions.

Neural value approximation provides a way to share statistical structure across routing contexts. In communication networks, You et al. [24] embed deep value networks in multi-agent Q-routing, allowing individual routers to estimate next-hop values from destination and richer traffic context. More generally, DQN represents action values with a deep neural network and stabilizes their learning through replay and target networks [25]. Double Q-learning separates action selection from action evaluation to reduce maximization bias [7], and Double DQN extends this separation to neural function approximation [26]. Within OHT routing, Ao et al. [27] encode the track map, vehicle state, and task information as a raster and train a convolutional map-information Double-DQN planner.

Conditioning one value function on multiple destinations is also related to universal value function approximation.

Universal value function approximators condition a shared value model on both the current state and a goal, enabling parameters to be shared across goal-dependent control problems [28]. This provides the general conceptual basis for a destination-conditioned shared value function. It does not, by itself, determine the routing decision interface, the available traffic information, or the manner in which the function should be initialized and updated in an OHT system.

Accordingly, the present work does not treat neural approximation, destination conditioning, or the Double-DQN target as individually new. Relative to per-node deep Q-routing and map-wide convolutional planning, its representation uses one fleet-wide state–action function to score each candidate outgoing edge from a fixed-dimensional feature vector. The same parameters are used across split nodes, destinations, vehicles, and task phases, while the existing controller continues to determine which outgoing edges are feasible. This parameter sharing is evaluated on the fixed fab guideway and should not be interpreted as zero-shot generalization to unseen layouts or destinations.

## D. Trajectory-Based Initialization and Online Adaptation

Previously collected control trajectories can be used to improve the initial behavior of an online value learner. Deep Q-learning from demonstrations pretrains a Q-network from demonstration data and continues learning online using temporal-difference, supervised action-classification, and prioritized replay objectives [29]. Its objective encourages the learned policy to reproduce demonstrated actions while retaining the ability to improve beyond the demonstrator. The present offline stage has a different role: it regresses returnto-go targets from trajectories generated by multiple routing policies and does not impose an imitation loss favoring the actions of a single demonstrator.

Offline reinforcement learning addresses the stronger problem of optimizing a policy or value function from a fixed dataset while controlling errors on poorly represented actions. Conservative value estimation and behavior-regularized policy improvement are representative approaches [30], [31]. Moving such a policy online introduces an additional distribution transition as newly collected experience begins to replace or supplement the offline data. Balanced replay with pessimistic estimates, policy expansion, calibrated value initialization, and periodic policy revitalization have been proposed to manage this transition in general control domains [32]–[35].

Neural Double Q-routing does not optimize a separate conservative offline policy. Its simulator-generated trajectories provide supervised return-to-go targets for initializing the same compact value function that is subsequently refined by online Double-DQN updates. The trajectories are collected from Dijkstra, Q-routing, and Double Q-routing so that the prior is not tied to a single behavior policy. Thus, the offline component is better viewed as trajectory-based value initialization than as a complete offline-RL stage.

Experience replay determines which online observations continue to influence the value model. Prioritized experience replay ranks individual transitions using temporal-difference error [36], whereas selective replay has considered criteria such as surprise, reward, distribution matching, and statespace coverage [37]. Event-stratified structured replay uses a domain-specific criterion instead: transitions are assigned to mutually exclusive OHT event classes, and update batches use predefined class quotas and loss weights. It therefore emphasizes observed congestion and service conditions rather than ranking samples by their current TD error, and it is not presented as an importance-sampling correction.

Online adaptation also requires responding to traffic changes that may be visible initially at only a small number of edges. Earlier predictive Q-routing modifies route estimates through stored favorable experience and predicted traffic recovery [22]. In the proposed controller, the shared neural value remains the recursively learned base estimate, while a recent edgeand task-phase-specific TD residual adjusts the current action ranking outside the bootstrap target. A count-dependent term is used only during the online warmup period. This separation allows recent local observations to affect immediate decisions without recursively propagating the local correction through the shared value function.

## E. Positioning of Neural Double Q-Routing

The closest methods can be distinguished along four dimensions. First, dispatching and rebalancing choose vehicles or idle-capacity locations, traffic predictors and link-weight controllers modify inputs to path construction, and routeguidance methods choose a next hop for an already assigned vehicle. Second, tabular Q-routing stores separate destination– node–neighbor values, whereas map- and graph-based methods use global spatial representations; the proposed model instead scores one candidate edge from bounded state–action features using parameters shared throughout the fixed guideway. Third, offline RL learns a policy from a fixed dataset, whereas the proposed offline data are used only to initialize the value function that continues to learn online. Fourth, generic prioritized replay ranks transitions by learning signals, whereas the proposed replay mechanism retains predefined OHT congestion and service-event classes.

Neural Double Q-routing therefore combines established reinforcement-learning components within an OHT-specific deployment architecture: a local controller-compatible nexthop decision, a shared candidate-edge value representation, mixed-trajectory initialization, Double-DQN refinement, and online mechanisms for recent and infrequent congestion evidence. This combination—rather than neural approximation, Double Q-learning, or offline pretraining in isolation—defines the methodological position of the work.

## III. OHT ROUTING PROBLEM FORMULATION

This section formalizes the OHT routing problem in three parts. First, we describe the directed guideway and define the split-based routing process. Second, we explain how learning is shared across the fleet while routing decisions are executed vehicle-wise, and specify the state and candidateaction representations used at each decision epoch. Finally, we characterize the variable-duration decision intervals, define the routing reward, and present the criteria used to evaluate controller performance.

## A. Guideway Model and Split-Based Routing Process

We model the OHT guideway as a directed graph $G =$ $( V , E )$ , where nodes represent load/unload access points, intersections, and geometric control points, and directed edges represent unidirectional track segments. The full map used in our experiments (Fig. 3) contains $\lvert V \rvert = 3 6 8 4$ nodes, $| E | =$ 4257 directed edges, and $| D | = 6 0 8$ distinct destinations. Its maximum out-degree is $\mathrm { d e g } _ { \mathrm { m a x } } ^ { + } = 2$ , meaning that every split is binary. A destination is a task-relevant endpoint node in $V$ at which a vehicle either picks up or drops off a carrier. In practice, these destinations correspond to load/unload ports at process tools, stockers, and buffers. The destination set $D \subset V$ is therefore determined by the fab layout rather than by the task stream.

Routing decisions are made at vehicle-specific split events. A decision occurs when a vehicle reaches a split node i satisfying $\deg ^ { + } ( i ) \geq 2$ , and the corresponding action selects one outgoing edge. After the action is selected, the vehicle follows any intervening degree-one segments until it reaches the next split or until the current task phase terminates at its pickup or delivery endpoint. Consequently, different routing decisions may span different numbers of physical edges and different amounts of elapsed time. We therefore model the routing process as semi-Markov [38].

Learning operates on the embedded sequence of routing decision epochs. Each routing choice contributes one discount step, whereas the realized holding time is incorporated into the interval reward defined in Eq. (3). The resulting value function ranks candidate actions under this decision-epoch discounted routing criterion.

## B. Fleet-Wide Learning and Decision Representation

Learning is shared across the fleet. Transitions collected from all vehicles enter common replay storage and are used to update a single value function $Q _ { \theta }$ . During execution, each vehicle applies this shared function to its controller-admitted outgoing edges using its own task state and bounded local traffic summaries. The resulting architecture combines centralized data aggregation and parameter learning with vehicle-wise local next-hop selection.

State and candidate-action features. At a split node i with target $d ,$ the state vector $s ~ \in ~ \mathbb { R } ^ { d _ { s } }$ , where $d _ { s } ~ = ~ 1 0$ contains the normalized indices of i and $d ,$ the normalized shortest-path distance $h ( i , d )$ , the in- and out-degrees of $i ,$ a load/unload-node indicator, a three-component task-phase code representing pickup, delivery, or other, and the vehicle’s recent congestion-waiting time.

The available actions are the graph out-neighbors of $i .$ For each candidate next hop $j ,$ the action vector $\phi _ { a } ( s , j ) \in$ $\mathbb { R } ^ { d _ { a } }$ , where $d _ { a } ~ = ~ 1 4$ , contains the normalized candidate index, edge length, remaining distance $h ( j , d )$ , signed progress $h ( i , d ) - h ( j , d )$ , candidate in- and out-degrees, edge occupancy, candidate-node queue length, and six local pressure summaries. These six summaries comprise bottleneck pressure, spillback risk, maximum and mean one-hop downstream pressure, maximum two-hop pressure, and the fraction of vehicles on the candidate edge whose waiting flag is active. All continuous components are scaled and clipped to [0, 1]. Appendix A-B specifies each feature dimension and its normalization.

![](images/701f263696ef50b86f295b9a20d3358c9dd9a4712b58b050861d8451fd35ac7a.jpg)  
Fig. 3. Guideway graph used in the simulation. Small markers indicate control nodes, including the degree-one nodes that preserve track geometry.

Local pressure summaries. More precisely, let $c _ { i j }$ denote the normalized occupancy of candidate edge $( i , j ) , \ q _ { j }$ the normalized queue at node $j , p _ { \mathrm { m a x } } ^ { ( 1 ) }$ and $p _ { \mathrm { m e a n } } ^ { ( 1 ) }$ the maximum and mean occupancies of the edges leaving $j ,$ and $p _ { \mathrm { m a x } } ^ { ( 2 ) }$ the maximum occupancy one additional hop downstream. The six local pressure summaries are

$$
\begin{array} { r l } & { \operatorname* { m a x } \{ c _ { i j } , p _ { \operatorname* { m a x } } ^ { ( 1 ) } , p _ { \operatorname* { m a x } } ^ { ( 2 ) } \} , \quad \operatorname* { m i n } \{ 1 , ( q _ { j } + p _ { \operatorname* { m a x } } ^ { ( 1 ) } ) / 2 \} , } \\ & { p _ { \operatorname* { m a x } } ^ { ( 1 ) } , \quad p _ { \operatorname* { m e a n } } ^ { ( 1 ) } , \quad p _ { \operatorname* { m a x } } ^ { ( 2 ) } , \quad n _ { i j } ^ { \mathrm { w a i t } } / \operatorname* { m a x } \{ 1 , n _ { i j } \} , } \end{array}\tag{1}
$$

where $n _ { i j }$ counts the vehicles on edge $( i , j )$ and $n _ { i j } ^ { \mathrm { w a i t } }$ counts those with an active waiting flag. Empty downstream sets contribute zero.

The features are assembled from the vehicle’s task state, the static map, and the simulator’s current edge-occupancy and node-queue records. Computing the one- and two-hop summaries requires information only from this bounded local neighborhood. Parameter updates, in contrast, aggregate transitions collected from the entire fleet.

## C. Decision-Interval Dynamics and Routing Criterion

Decision-interval dynamics. Let $\mathcal { T } _ { t }$ denote the ordered sequence of physical edges traversed during decision interval t. The elapsed time of the interval is

$$
\tau _ { t } = \sum _ { e \in \mathcal { T } _ { t } } \left( m _ { t , e } + w _ { t , e } + b _ { t , e } \right) ,\tag{2}
$$

where $m _ { t , e }$ is the time spent moving on edge $e , w _ { t , e }$ is the time spent waiting for separation or right-of-way control, and $b _ { t , e }$ is the time blocked by downstream conditions or resourceadmission constraints. For an individual vehicle, $w _ { t , e }$ and $b _ { t , e }$ depend on the motion and service states of the rest of the fleet. The transition to $s _ { t + 1 }$ is therefore stochastic even when the map and selected candidate edge are fixed.

Routing reward. The reward accumulates the realized time and deadlock warnings over the physical edges traversed during the interval. Risk and route progress are evaluated once for the action selected at the initial split:

$$
\begin{array} { l } { r _ { t } = - \displaystyle \sum _ { e \in \mathbb { Z } _ { t } } ( w _ { m } m _ { t , e } + w _ { w } w _ { t , e } + w _ { b } b _ { t , e } + w _ { d } \omega _ { t , e } ) } \\ { \quad \quad \quad - w _ { r } \rho ( s _ { t } , a _ { t } ) + w _ { p } \delta _ { h } ( s _ { t } , a _ { t } ) , } \end{array}\tag{3}
$$

where $\rho ( s _ { t } , a _ { t } )$ is the maximum of the selected-edge occupancy, candidate-node queue, bottleneck pressure, spillback risk, and maximum one-hop downstream pressure. The term $\omega _ { t , e }$ is the deadlock-warning level recorded on physical edge e. Route progress is defined as

$$
\delta _ { h } ( s _ { t } , a _ { t } ) = \mathrm { c l i p } \left( \frac { h ( i , d ) - h ( j , d ) } { D _ { s } } , - 1 , 1 \right) ,
$$

where $D _ { s }$ is the distance scale. The corresponding action feature is mapped affinely to [0, 1], whereas the reward retains the signed progress value.

Routing criterion and parameters. The reward defines a domain-weighted routing criterion. Its time-based terms retain physical units while assigning different costs to movement, waiting, and blocking. The risk, warning, and progress terms provide anticipatory signals for congestion and route advancement. These shaping terms are not constructed as potential differences [39]; instead, they are components of the learned routing criterion rather than a policy-invariant transformation of elapsed time.

We fix

$$
( w _ { m } , w _ { w } , w _ { b } , w _ { r } , w _ { d } , w _ { p } ) = ( 1 . 0 , 0 . 8 , 1 . 2 , 0 . 1 , 0 . 1 5 , 0 . 3 )
$$

and use warning levels $\omega ~ \in ~ \{ 0 , 1 , 3 \}$ in every cell. $\mathsf { A p - }$   
pendix A-D2 provides the remaining threshold definitions.

Evaluation criteria. Controller performance over a wallclock horizon of $T = 1 0 0 0 s$ is evaluated using mean completion time (CT Mean). We additionally report the number of completed tasks (Completed) and the 95th-percentile completion time (CT P95) to identify policies that either leave tasks unfinished or shift delays into the tail of the completion-time distribution. Startup analyses use an additional early-horizon window.

The resulting formulation specifies the decision process, input representation, transition dynamics, and routing criterion used by the offline-to-online Neural Double Q-routing method developed in Section IV.

## IV. OFFLINE-TO-ONLINE NEURAL DOUBLE Q-ROUTING

This section develops Neural Double Q-routing in four parts. We first define the shared state–action neural value function and then describe its offline warm start from mixed routing trajectories. Next, we present the online Double-Q refinement rule. Finally, we detail the local congestion correction and event-stratified structured replay used during online adaptation. Fig. 4 summarizes the complete flow from offline initialization to online action selection and learning. We refer to the complete method as Neural Double Qrouting; tables and figures use the compact implementation label QNeuralDouble.

![](images/1e134559a5c8e12834175c6c55118d333c3ae1a448a2771449cd579e5d658593.jpg)  
Fig. 4. Method overview. Logged trajectories initialize the online and target value networks. During operation, Double-Q updates refine the shared value function, while a local congestion correction adjusts action rankings from recent edge-level observations and event-stratified structured replay retains predefined congestion and service-event classes.

## A. Shared State–Action Neural Value Function

Tabular Q-routing stores an independent value $Q ( d , i , j )$ for every destination-node-action tuple. When the guideway has thousands of nodes and hundreds of destinations, most tuples are visited infrequently, and information learned at one tuple does not transfer to structurally similar situations elsewhere on the map.

We replace this table with a neural value function that scores each candidate next hop through a shared parameterization. For a vehicle in state s considering action a, we construct a joint feature vector $\phi ( s , a )$ and predict:

$$
Q _ { \theta } ( s , a ) = f _ { \theta } { \bigl ( } \phi ( s , a ) { \bigr ) } ,\tag{4}
$$

where $f _ { \theta }$ is a two-hidden-layer MLP with 64 units per layer and ReLU activations. The state $( d _ { s } = 1 0 )$ and action $( d _ { a } =$ 14) features are exactly those defined in Section III, giving joint input dimension $d _ { \phi } = 2 4$ ; all inputs are pre-normalized to comparable scales. The scalar output approximates the discounted return-to-go under the reward in Eq. (3); actions with larger values are therefore preferred. At each junction the router first evaluates $Q _ { \theta } ( s , a )$ for every feasible next hop. Section IV-D defines the local correction applied to these base values before the final action is selected. Including biases, the MLP has 5,825 trainable parameters versus ∼2.59M entries in tabular Q on our guideway (Appendix A-A).

Because θ is shared across nodes and destinations, every update changes one scoring function used throughout the map rather than one isolated table entry. Appendix A-C examines the resulting estimate stability; it does not treat lower variance as evidence of value accuracy.

## B. Offline Return-to-Go Warm Start

Random initialization leaves the value model uninformative at the start of operation. We instead pretrain $f _ { \theta }$ on logged decision-interval trajectories generated by Dijkstra, Q-routing, and Double Q-routing.

Each record contains $\left( { { s _ { t } } , { a _ { t } } , { r _ { t } } , { s _ { t + 1 } } , { z _ { t } } } \right)$ , where $z _ { t }$ marks a terminal task-phase boundary. Within a trajectory segment ending at index $T _ { \ell } .$ , the regression target is

$$
\hat { G } _ { t } = \sum _ { k = t } ^ { T _ { \ell } } \gamma ^ { k - t } r _ { k } .\tag{5}
$$

The accumulation is reset at changes in episode, vehicle, task, target, task phase, or terminal segment, so rewards from distinct routing episodes are not combined. We fit the prior by minimizing

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r i o r } } ( \theta ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \Big [ \big ( Q _ { \theta } ( s , a ) - \hat { G } ( s , a ) \big ) ^ { 2 } \Big ] . } \end{array}\tag{6}
$$

The learned weights initialize both online $Q _ { \theta }$ and target $Q _ { \bar { \theta } }$ Training batches balance the logged sources by operating cell, behavior policy, and collection seed so that a large stratum does not dominate the regression. Each of the three behavior policies is collected with three seeds. The prior is trained for 40 epochs with Adam at learning rate $1 0 ^ { - 3 }$ and batch size $6 4 ;$ Appendix A-D1 gives the dataset construction details.

## C. Online Double-Q Refinement

Standard Q-learning’s max ${ } _ { a ^ { \prime } } Q _ { \theta } ( s ^ { \prime } , a ^ { \prime } )$ bootstrap systematically overestimates noisy values—amplified in OHT routing where candidate hops often have similar true values and sparse visit counts. We use $\gamma ~ = ~ 0 . 9 9$ on the embedded decision chain described in Section III; interval duration is already represented in r through Eq. (3). The Double-DQN form of the Double-Q idea [7], [26] separates action selection from value evaluation:

$$
a ^ { \star } = \arg \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta } ( s ^ { \prime } , a ^ { \prime } ) , \qquad y = r + \gamma ( 1 - z ) Q _ { \bar { \theta } } ( s ^ { \prime } , a ^ { \star } ) ,\tag{7}
$$

where $z = 1$ for a terminal transition; in that case $y = r$ and no next action is evaluated. The online network is updated by minimizing the Huber loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t d } } ( \theta ) = \mathbb { E } \left[ \ell _ { \mathrm { H u b e r } } ( Q _ { \theta } ( s , a ) - y ) \right] . } \end{array}\tag{8}
$$

Target networks stabilize DQN-style learning [25], [26]. Here $Q _ { \bar { \theta } }$ is updated by Polyak averaging $\bar { \theta }  ( 1 - \tau ) \bar { \theta } + \tau \theta$ with $\tau = 0 . 0 0 5$ . Online updates use Adam at learning rate $1 0 ^ { - 3 }$ batch size 64, and gradient-norm clipping at 5. Updates begin after 64 replay records and are performed every four observed transitions.

## D. Congestion-Aware Online Adaptation

Local congestion correction. The shared value model can lag behind a short-lived local bottleneck. We therefore combine its raw return estimate with a local TD-residual estimate and a count-based uncertainty penalty:

$$
\begin{array} { r } { U _ { t } ( s , a ) = Q _ { \theta } ( s , a ) + \lambda _ { t } \kappa ( s , a ) R _ { \mathrm { l o c a l } } ( s , a ) } \\ { - \beta _ { t } ( n ( s , a ) + 1 ) ^ { - 1 / 2 } , \qquad } \end{array}\tag{9}
$$

where $R _ { \mathrm { l o c a l } }$ is an exponentially averaged, clipped TD residual in the same reward units as $Q _ { \theta }$ , and $\kappa ( s , a )$ downweights an edge-level residual carried across task phases. Here $n ( s , a )$ is the observation count for the corresponding directed edge and task phase; an edge-level fallback uses the count pooled across phases. The schedules $\lambda _ { t }$ and $\beta _ { t }$ increase the role of observed local residuals and remove the initial uncertainty penalty over $N _ { \mathrm { w a r m u p } }$ transitions. For candidate edge and phase pair $( e , p )$ the residual update is

$$
R _ { e , p } \gets ( 1 - \alpha ) R _ { e , p } + \alpha \ \mathrm { c l i p } ( y _ { t } - Q _ { \theta } ( s , a ) , - c , c ) ,\tag{10}
$$

with $\alpha \ = \ 0 . 0 5$ and $c = 5$ . We set $\kappa ~ = ~ 1$ for an exact edge–phase record, 0.5 for an edge-level record pooled across phases, and 0 when no record exists. The schedules are

$$
\lambda _ { t } = 0 . 5 \operatorname* { m i n } ( t / 5 0 0 0 , 1 ) , \qquad \beta _ { t } = 0 . 3 \operatorname* { m a x } ( 1 - t / 5 0 0 0 , 0 ) .\tag{11}
$$

The executed action is $a _ { t } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } ( s _ { t } ) } U _ { t } \big ( s _ { t } , a \big )$ . The two quantities have separate roles: $Q _ { \theta }$ is the recursively trained base return estimator, whereas $U _ { t }$ is the action score used by the behavior policy at execution. Transitions generated by that policy enter replay, and $Q _ { \theta }$ is updated off-policy using the base Double-Q target in Eq. (7). Keeping the edge-level residual outside the bootstrap lets recent observations alter the current action ranking without recursively propagating those residuals through future targets.

Event-stratified structured replay (SR). Uniform sampling can make transitions involving blocking or prolonged waiting uncommon in an update batch. We assign each transition exactly one of five labels—normal, congested, blocked, near-deadlock, or load/unload (LU) bottleneck—using an explicit severity order. Non-normal transitions are also retained in a recent event pool. Each update draws a configured share from this pool, allocates that share across event labels, and applies fixed class-dependent loss weights. In the reported configuration, event records target 40% of a batch, giving nominal whole-batch shares of $1 6 / 1 2 / 8 / 4 \%$ for congested, blocked, near-deadlock, and LU-bottleneck records; normal records fill the remaining 60%. This is domain-informed oversampling with a cost-sensitive loss, not an importance sampling correction. Unlike PER [36], which ranks individual samples by TD error, the event labels are determined from observed OHT traffic and service conditions. Overlapping conditions are resolved by

TABLE I  
STRUCTURED-REPLAY LABELS AND NOMINAL WHOLE-BATCH SHARES WHEN ALL CLASSES ARE AVAILABLE. WEIGHT MULTIPLIES TD LOSS.
<table><tr><td>Label</td><td>Trigger</td><td></td><td>Share Weight</td></tr><tr><td>normal</td><td> $_ \mathrm { o t h e r w i s e }$ </td><td>60%</td><td>1.0</td></tr><tr><td>congested</td><td> $w _ { t } \geq 1 . 5 \mathrm { s }$ </td><td>16%</td><td>1.5</td></tr><tr><td>blocked</td><td> $b _ { t } > 0$ </td><td>12%</td><td>2.0</td></tr><tr><td>near-deadlock</td><td> $\omega _ { t } = 3$ </td><td>8%</td><td>3.0</td></tr><tr><td></td><td>LU-bottleneck at load/unload node and  $w _ { t } > 1 5 \mathrm { s }$ </td><td>4%</td><td>2.0</td></tr></table>

$$
\begin{array} { r } { \begin{array} { r } { \mathrm { n e a r - d e a d l o c k } \succ \mathrm { L U - b o t t l e n e c k } \succ \mathrm { b l o c k e d } } \\ { \qquad \succ \mathrm { c o n g e s t e d } \succ \mathrm { n o r m a l } . } \end{array} } \end{array}\tag{12}
$$

Table I gives the triggers and loss weights; missing event classes are filled first from other event records and then from general replay.

## V. EXPERIMENTAL EVALUATION

This section first specifies the simulation setup, baselines, and matched-scene protocol. It then reports the long-horizon results across the fleet-size and arrival-rate sweep, and finally presents the warm-start and online-component ablations. Within each operating cell, every method receives the same task sequence and initial conditions.

## A. Setup

All methods are evaluated on the fab guideway of Section III $\left( \lvert V \rvert = 3 6 8 4 , \lvert E \rvert = 4 2 5 7 , \lvert D \rvert = 6 0 8 \right)$ under a common discreteevent simulator. We sweep fleet $\mathrm { s i z e } ~ \in ~ \{ 1 0 0 , 1 5 0 , 2 0 0 \}$ and task arrival rate $( \mathbf { A R } ) \in \{ 1 . 0 , 1 . 5 , 2 . 0 \}$ task $s ^ { - 1 }$ , giving 3×3 cells. Within each cell every method shares the same task stream, initial vehicle placement, and random seed. The simulation horizon is $T = { 1 0 0 0 } \mathrm { s } ,$ and startup analyses use the first 200 s. We report mean completion time (CT Mean), completed tasks, and 95th-percentile completion time (CT P95). In the warm-start study, Early CT is the mean CT of tasks completed within the first 200 s.

Baselines. Dijkstra ignores dynamics; tabular Qrouting [2], [6] adapts to congestion without bias correction; tabular Double Q-routing [7] adds overestimation control. QNeuralDouble (ours) uses the full framework: offline prior, neural value function, online Double-Q updates, local correction, and structured replay.

## B. Long-horizon results across the sweep

Table II reports matched-scene results over the 1000 s horizon for all nine cells. Bold marks the best method per cell; ∆ is Ours vs. QDouble (negative is improvement).

Relative to QDouble, QNeuralDouble reduces CT Mean in every cell by 0.8%–8.8%; completed-task counts stay within 1% of the baseline in all but one cell. At 150 OHTs the CT Mean reductions are 3.4%–8.8%, and CT P95 improves by 3.4%–6.2%. Fig. 6 visualizes the full CT Mean sweep.

(a)  
TABLE II  
MATCHED-SCENE 1000 s LONG-HORIZON RESULTS ACROSS THE FLEET × ARRIVAL-RATE SWEEP. VALUES ARE MEAN ± STD OVER 5 SEEDS.
<table><tr><td></td><td></td><td colspan="5">CT Mean (s)</td><td colspan="2">∆ CT Mean of Ours</td><td colspan="3">Other metrics of Ours</td></tr><tr><td>OHT</td><td>AR</td><td></td><td>D</td><td>Q</td><td>QDouble</td><td>Ours</td><td>vs QDouble (s)</td><td>∆ %</td><td>Compl.</td><td>CT P95 (s)</td><td>∆ P95</td></tr><tr><td>100</td><td>1.0</td><td> $\mathbf { 3 1 9 . 8 4 } \pm \mathbf { 3 . 4 6 }$ </td><td> $3 2 6 . 4 1 \pm 4 . 3 1$ </td><td></td><td> $3 2 4 . 1 8 \pm 3 . 2 1$ </td><td> $3 2 1 . 0 6 \pm 2 . 8 6$ </td><td>-3.12</td><td>-1.0%</td><td> $3 6 3 . 0 \pm 3 . 2$ </td><td> $5 9 4 . 2 1 \pm 9 . 5 0$ </td><td>-0.3%</td></tr><tr><td>100</td><td>1.5</td><td> ${ \bf 3 7 4 . 6 2 \pm 4 . 1 7 }$ </td><td> $3 7 9 . 9 5 \pm 4 . 9 4$ </td><td></td><td> $3 8 2 . 4 7 \pm 3 . 8 3$ </td><td> $3 7 7 . 8 6 \pm 3 . 3 2$ </td><td> $- 4 . 6 1 \quad - 1 . 2 \%$ </td><td></td><td> $3 8 5 . 0 \pm 2 . 6$ </td><td> $7 1 2 . 8 8 \pm 1 1 . 3 6$ </td><td>+0.2%</td></tr><tr><td>100</td><td>2.0</td><td> ${ \bf 4 1 3 . 0 5 \pm 4 . 5 5 }$ </td><td></td><td> $4 1 6 . 7 8 \pm 5 . 4 7$ </td><td> $4 1 8 . 0 4 \pm 4 . 2 3$ </td><td> $4 1 4 . 2 7 \pm 3 . 6 7$ </td><td>-3.77</td><td>-0.9%</td><td> $3 7 4 . 0 \pm 4 . 2$ </td><td> $7 6 5 . 4 0 \pm 1 2 . 3 0$ </td><td>-0.5%</td></tr><tr><td>150</td><td>1.0</td><td> $2 0 5 . 9 1 \pm 4 . 8 8$ </td><td></td><td> $1 9 7 . 1 8 \pm 4 . 3 7$ </td><td> $1 9 5 . 0 6 \pm 3 . 5 4$ </td><td> $\mathbf { 1 7 7 . 8 4 } \pm \mathbf { 2 . 0 7 }$ </td><td>-17.22</td><td>-8.8%</td><td> $5 8 8 . 0 \pm 8 . 1$ </td><td> $3 6 2 . 7 7 \pm 8 . 3 0$ </td><td>-6.2%</td></tr><tr><td>150</td><td>1.5</td><td> $1 8 7 . 3 0 \pm 4 . 5 0$ </td><td></td><td> $1 7 6 . 8 5 \pm 3 . 8 9$ </td><td> $1 7 4 . 4 2 \pm 3 . 2 2$ </td><td> ${ \bf 1 6 8 . 5 0 \pm 2 . 1 2 }$ </td><td>-5.92</td><td>-3.4%</td><td> $9 1 6 . 2 \pm 1 2 . 7$ </td><td> $3 3 7 . 0 4 \pm 7 . 7 5$ </td><td>-3.9%</td></tr><tr><td>150</td><td>2.0</td><td> $2 0 0 . 0 4 \pm 4 . 7 5$ </td><td></td><td> $1 8 8 . 2 7 \pm 4 . 1 2$ </td><td> $1 8 5 . 6 9 \pm 3 . 4 0$ </td><td> ${ \bf 1 7 4 . 8 6 \pm 2 . 2 3 }$ </td><td>-10.83</td><td>-5.8%</td><td> $1 1 8 2 . 2 \pm 1 6 . 5$ </td><td> $3 6 4 . 1 8 \pm 8 . 4 4$ </td><td>-3.4%</td></tr><tr><td>200</td><td>1.0</td><td> $1 2 1 . 2 7 \pm 1 . 9 7$ </td><td></td><td> $1 2 0 . 4 3 \pm 2 . 0 2$ </td><td> $1 1 8 . 7 4 \pm 1 . 6 6$ </td><td> ${ \bf 1 1 7 . 2 0 \pm 1 . 5 4 }$ </td><td> $- 1 . 5 4 \quad - 1 . 3 \%$ </td><td></td><td> $5 4 4 . 0 \pm 5 . 4$ </td><td> $2 6 3 . 1 8 \pm 4 . 9 9$ </td><td>-0.6%</td></tr><tr><td>200</td><td>1.5</td><td> $1 4 1 . 1 8 \pm 2 . 5 4$ </td><td></td><td> $1 3 8 . 4 2 \pm 2 . 2 7$ </td><td> $1 3 6 . 8 5 \pm 1 . 8 8$ </td><td> ${ \bf 1 3 5 . 7 1 \pm 1 . 5 0 }$ </td><td> $- 1 . 1 4 \mathrm { ~ \ } - 0 . 8 \%$ </td><td></td><td> $8 4 7 . 8 \pm 8 . 4$ </td><td> $2 8 1 . 9 3 \pm 5 . 4 1$ </td><td>-0.9%</td></tr><tr><td>200</td><td>2.0</td><td> $2 3 5 . 8 6 \pm 4 . 2 7$ </td><td></td><td> $2 2 3 . 4 0 \pm 3 . 7 4$ </td><td> $2 1 3 . 7 9 \pm 3 . 0 4$ </td><td> ${ \bf 2 0 3 . 6 2 \pm 2 . 1 7 }$ </td><td> $- 1 0 . 1 7 \quad - 4 . 8 \%$ </td><td></td><td> $1 0 4 5 . 0 \pm 1 0 . 3$ </td><td> $3 7 6 . 6 1 \pm 7 . 1 9$ </td><td>-1.3%</td></tr></table>

TABLE III

WARM-START COMPARISON ON MATCHED STARTUP SCENES (FIRST 200 S, AR=1.0). EARLY CT IS THE MEAN CT OF TASKS COMPLETED WITHIN THIS INTERVAL. VALUES ARE MEAN ± STD OVER 5 SEEDS.
<table><tr><td>Scen.</td><td>Variant</td><td>Early CT</td><td> $\mathbf { C o m p l . }$ </td><td> $\mathrm { C T \ M e a n }$ </td><td>CT P95</td><td>∆ Compl.</td><td>∆ P95</td></tr><tr><td>A (150)</td><td>Cold</td><td> $5 1 . 0 8 \pm 1 . 4 3$ </td><td> $2 7 \pm 0 . 8$ </td><td> $5 9 . 0 4 \pm 1 . 2 0$ </td><td> $9 2 . 3 6 \pm 3 . 1 9$ </td><td>(ref.)</td><td>(ref.)</td></tr><tr><td>A (150)</td><td>Warm</td><td> ${ \bf 5 0 . 4 2 \pm 1 . 2 6 }$ </td><td> $3 3 \pm 1 . 8$ </td><td> ${ \pm 7 . 6 2 \pm 1 . 3 5 }$ </td><td> ${ \bf 8 5 . 7 1 } \pm 3 . 0 8$ </td><td>+22.2%</td><td>-7.2%</td></tr><tr><td>B (200)</td><td>Cold</td><td> $5 4 . 1 7 \pm 1 . 4 0$ </td><td> $2 6 \pm 1 . 4$ </td><td> $6 1 . 9 3 \pm 1 . 5 3$ </td><td> $1 1 0 . 8 4 \pm 3 . 8 3$ </td><td>(ref.)</td><td>(ref.)</td></tr><tr><td>B (200)</td><td>Warm</td><td> ${ \bf 5 3 . 0 5 \pm 1 . 1 8 }$ </td><td> $3 2 \pm 1 . 6$ </td><td> ${ \bf 6 0 . 7 4 \pm 1 . 2 8 }$ </td><td> ${ \bf 9 4 . 1 8 \pm 3 . 2 5 }$ </td><td>+23.1%</td><td>-15.0%</td></tr></table>

TABLE IV

COMPONENT AND BASELINE COMPARISONS AT 150 OHT, $\mathrm { A R } { = } 1 . 0$ (MEAN OVER 5 SEEDS). ∆ IS THE CT MEAN DIFFERENCE FROM THE FULLMODEL (POSITIVE IS WORSE). LC: LOCAL CONGESTION CORRECTION.SR: STRUCTURED REPLAY.
<table><tr><td>Variant</td><td>CT Mean</td><td>Compl.</td><td>CT P95</td><td>∆ CT Mean</td></tr><tr><td>QNeuralDouble (Full)</td><td>177.84</td><td>588</td><td>362.77</td><td>(ref.)</td></tr><tr><td>– w/o Structured Replay</td><td>179.45</td><td>587</td><td>372.84</td><td>+1.61</td></tr><tr><td>— w/o Local Correction</td><td>182.71</td><td>585</td><td>384.06</td><td>+4.87</td></tr><tr><td>– w/o Double-Q (Neural Single-Q)</td><td>183.62</td><td>586</td><td>386.94</td><td>+5.78</td></tr><tr><td>− w/o Both (LC + SR)</td><td>185.16</td><td>586</td><td>379.42</td><td>+7.32</td></tr><tr><td>Tabular Double-Q baseline</td><td>195.06</td><td>592</td><td>386.75</td><td>+17.22</td></tr></table>

## C. Ablations

The warm-start study uses the 150- and 200-OHT scenarios in Table III; the component study uses the 150-OHT, AR= 1.0 cell in Table IV.

Table III compares warm-started and cold-started QNeural-Double. Early CT, defined as the mean CT of tasks completed within the first 200 s, improves by 1.3% (Scen. A) and 2.1% (Scen. B). Over the same startup window, the warm-started router completes 22.2% and 23.1% more tasks, while CT P95 decreases by 7.2% and 15.0%. Fig. 5(a) visualizes these relative changes.

For the online-component study, we use the 150 OHT, $\mathrm { { A R } = 1 . 0 }$ cell. Starting from the full model, we separately disable structured replay (SR; unit loss weights and no eventstratified sampling), local correction (LC; $\lambda _ { t } ~ \equiv ~ \beta _ { t } ~ \equiv ~ 0 )$ Double-Q (single estimator), and both LC+SR. Offline warm start and the matched-scene protocol are fixed for these neural component deletions. The final row is the ordinary tabular Double Q-routing baseline and is therefore a framework-level comparison rather than a single-component ablation. Table IV reports the results.

![](images/96fa0613eae56794538de692a1253e361b9186c9deac9923cca13d1cdd3f0b05.jpg)

Among the neural component deletions, replacing Double-Q with a single estimator produces the largest mean CT penalty (+5.78 s), followed by removing local correction (+4.87 s) and structured replay (+1.61 s). Removing LC and SR together gives a +7.32 s penalty, so their observed effects are not additive in this cell. The full framework differs from the tabular Double-Q baseline by 17.22 s; because this comparison changes the representation together with the initialization and adaptation mechanisms supported by it, the difference is an overall framework comparison and is not attributed to the neural representation alone. Fig. 5(b) visualizes the CT Mean comparisons.

![](images/a66532213efbcaea8b897141243029197a781cf82648d06edc26ba1355b88ac0.jpg)  
Fig. 5. Warm-start and component analyses. (a) Relative warm-start improvement over cold start during the first 200 s; Early CT is the mean CT of tasks completed in that interval. (b) CT Mean for the neural component deletions and the tabular Double-Q baseline; parentheses report the difference from the full framework. The panels visualize Tables III and IV, respectively.

## VI. DISCUSSION AND LIMITATIONS

In the selected ablation cell, each online component has a positive mean contribution, although the effects of local correction and structured replay are not additive (Table IV). The tabular Double-Q row changes the representation, initializa tion, and adaptation design together, so its larger gap measures the complete framework rather than the neural representation alone. The warm-start study addresses a different question:

CT Mean across fleet size and task arrival rate (lower is better)

![](images/9660f63842990b4391f0979f7c8a02a08fb67edbbad24d7449c02f84c9a9d690.jpg)  
Fig. 6. CT Mean across the fleet-size and arrival-rate sweep, visualizing Table II. Red annotations report the relative change of QNeuralDouble from tabular Double Q-routing; negative values indicate lower CT Mean.

Early CT changes modestly, while completed-task count and CT P95 change more clearly during the first 200 s (Table III).

The method ranking changes with both fleet size and task arrival rate. Dijkstra obtains the lowest CT Mean in all three 100-OHT cells, whereas QNeuralDouble performs best in all 150- and 200-OHT cells. Relative to tabular Double Q-routing, the largest reductions occur at 150 OHTs, ranging from 3.4% to 8.8%. At 200 OHTs, the reductions are 1.3%, 0.8%, and 4.8% at AR= 1.0, 1.5, and 2.0, respectively, with the largest reduction at AR= 2.0. The evidence therefore supports a setting-dependent benefit rather than a monotone fleet-size or arrival-rate effect. Because utilization and congestion-source statistics are not reported, we do not attribute these patterns to a specific capacity regime.

Limitations. The evaluation is on a single fab layout within one simulator and not validated on a physical OHT system. The external baselines are tabular policies; we include a neural single-Q variant as an internal ablation but do not compare against GNN routers (e.g., HarmonyRouting [18], for which a public implementation is unavailable) or multi-agent RL baselines. The offline prior mixes three reasonable but suboptimal policies, and we have not characterized robustness to substantially worse or adversarial logged data. We fix the task-dispatching rule and study routing in isolation, though dispatching and routing interact. The learning criterion uses decision-epoch discounting and domain-weighted shaping. Future work will study duration-aware discounting, potentialbased shaping, and sensitivity to the reward weights and deadlock-warning thresholds in Appendix A-D2.

## VII. CONCLUSION

We presented a neural Double Q-routing framework for large-scale OHT systems. It replaces classical Q-routing’s destination-indexed table with a shared state-action value function, initializes that function from mixed Dijkstra/Q/Double-Q trajectories, and refines it online with Double-Q updates, local congestion correction, and structured replay. Across matched simulations spanning 100/150/200 OHTs and 1.0/1.5/2.0 task s<sup>−1</sup> arrival rates, the framework improves CT Mean over tabular Double Q-routing in every cell (up to 8.8%), while completed-task counts remain within 1% in eight of nine cells.

The ablations qualify this aggregate result. In the selected cell, each online component has a positive mean contribution, and the complete framework also outperforms the tabular Double-Q baseline. Offline initialization matters most during startup: it raises the number of tasks completed during the first 200 s by up to 23% and reduces startup tail latency by up to 15%.

## APPENDIX A

## ADDITIONAL ANALYSES AND IMPLEMENTATION DETAILS

This appendix is organized from general analysis to reproducibility details. It first compares the computational complexity of tabular and neural Q-routing, then provides the complete state and action feature definitions and examines value-estimate stability. Finally, it records the implementation settings for offline pretraining, reward design, local correction, and replay storage.

## A. Complexity analysis

Space complexity. The tabular Q-routing methods maintain a three-element tuple $( d , i , j )$ representing the destination, current node, and next hop. Let |D| denote the number of distinct destinations, |V| the number of nodes, and |E| the number of directed edges. The table requires $\begin{array} { r l } { \sum _ { i \in V } \left| D \right| \cdot \mathrm { d e g } ^ { + } ( i ) = } & { { } } \end{array}$ $| D | \cdot | E |$ scalar entries. In our fab guideway, $\vert V \vert = 3 6 8 4 ,$ $| E | \ = \ 4 2 5 7 .$ and $| D | \ = \ 6 0 8$ yield approximately 2.59M independent parameters that must each be visited and updated separately. Double Q-routing doubles this to around 5.18M entries.

Our neural value function replaces the entire table with a two-hidden-layer MLP. With input dimension $d _ { \phi }$ (the joint state-action feature size), hidden size $h ,$ and a single scalar output, the parameter count is $d _ { \phi } h + h + h ^ { 2 } + h + h + 1 =$ $O ( d _ { \phi } h + h ^ { 2 } )$ , including the three bias vectors. This count is independent of the destination set size $| D |$ , the number of nodes $| V |$ , and the number of directed edges |E|. With $d _ { \phi } = 2 4$ and $h = 6 4$ , the network has $1 5 3 6 + 6 4 + 4 0 9 6 + 6 4 + 6 4 + 1 = 5 8 2 5$ parameters. This is about $4 4 5 \times$ fewer parameters than the tabular baseline has entries and $8 8 9 \times$ fewer than tabular Double Q. The neural model scales as $O ( d _ { \phi } h + h ^ { 2 } )$ , whereas the tabular storage grows as $O ( | D | | E | )$

Time complexity. At each junction with k feasible next hops, the tabular router performs k table lookups at $O ( 1 )$ each, giving $O ( k )$ per decision. The neural router evaluates a forward pass for each candidate, costing $O ( k ( d _ { \phi } h + h ^ { 2 } ) )$ In our guideway, the maximum out-degree is two as all split nodes are binary, so each decision requires at most two forward passes of about 5800 multiply-adds.

## B. Full state and action feature definitions

State features $( d _ { s } = 1 0 )$ . The state s observed by a vehicle at a decision node i with target d contains: (i) the current and target node identifiers encoded as normalized scalar indices $\operatorname { i d x } ( i ) / ( | V | - 1 )$ and $\operatorname { i d x } ( d ) / ( | V | - 1 )$ , which provide a coarse positional cue without inflating the input dimension with onehot vectors over $\vert V \vert ~ = ~ 3 6 8 4$ nodes; (ii) the shortest-path distance $h ( i , d )$ , normalized by $D _ { s } = \bar { w } \sqrt { | V | }$ , where w¯ is the mean edge weight; (iii) the in-degree and out-degree of $i ,$ each normalized by the maximum degree in G; (iv) a load/unloadnode indicator that is 1 when i is associated with a task pickup or drop-off point and 0 otherwise; (v) a one-hot encoding of the task phase (pickup, delivery, or other); and (vi) the recent waiting time of the vehicle, normalized by a fixed reference of 5 s. All continuous components lie in [0, 1] after normalization, and all binary components take values in $\{ 0 , 1 \}$ , so that no single feature dominates the input scale.

Action features $( d _ { a } ~ = ~ 1 4 )$ . For each candidate next hop $j ,$ the action vector contains: (i) the normalized scalar index of $j ; ~ ( \mathrm { i i } )$ the normalized length of edge $( i , j )$ ; (iii) the normalized remaining distance $h ( j , d )$ ; (iv) the progress $h ( i , d ) - h ( j , d )$ , affinely mapped to $[ 0 , 1 ] ; ( \mathrm { v } )$ the normalized in- and out-degrees of $j ; \mathsf { \Gamma } ( \mathrm { v i } )$ the occupancy of $( i , j )$ relative to its length-based capacity; (vii) the waiting-queue length at $j ,$ normalized by the same fixed five-vehicle reference used by the implementation; and (viii) the following six pressure summaries.

## C. Value-estimate stability of tabular vs. neural Q-routing

For both routers, $\sigma ( \hat { Q } )$ measures how much a value estimate fluctuates near convergence. During the final training window, we take periodic snapshots of $\hat { Q } ( d , i , j )$ for every tuple and report the standard deviation over the last few snapshots. Fig. 7 shows higher tabular variation across most of the guideway and lower variation for the shared neural value function. This pattern is consistent with parameter sharing across routing contexts, but the figure measures stability rather than value accuracy; routing outcomes are reported in Section V.

![](images/3916828a4be11e4df16889bd7a32ef9f65e2de24cf405bb46cfd7c1845ef510d.jpg)  
Fig. 7. Value-estimate stability of tabular and neural Q-routing on the fab guideway of Fig. 3. All panels use the same layout, with color aggregated by bay. (a) Visit frequency of $( d , i , j )$ tuples. (b) Standard deviation of tabular estimates over the final snapshots. (c) Standard deviation of neural estimates over the same snapshots.

## D. Implementation Details

The reported configuration is held fixed across the fleet-byarrival-rate sweep.

1) Offline pretraining and dataset construction: Dataset composition. The offline data come from the three behavior policies and three collection seeds specified in Section IV. Strata indexed by operating cell, policy, and seed are sampled equally so that larger strata do not dominate through sample count. Return-to-go accumulation uses the boundaries in Eq. (5). The online general and event replay stores contain 5,000 and 2,000 transitions, respectively, and both discard the oldest record when full.

2) Reward design and weights: The reward in $\operatorname { E q . } \ ( 3 )$ has three groups of terms: the per-edge realized-time penalties summed over $e \in \ \mathcal { T } _ { t }$ , the split-level local risk and routeprogress terms, and the per-edge warning penalty. The time terms set the principal scale, with separate weights for motion, waiting, and blocking. Risk and progress are smaller anticipatory terms in the same training criterion.

The weight tuple is reported after Eq. (3); movement time defines the reference scale, waiting and blocking remain on the same scale, and the risk and progress terms are smaller shaping terms. The warning proxy takes the three values 0, 1, and 3. Warning is triggered when waiting exceeds 3 s with an immediate downstream queue of at least two; it becomes critical after the warning condition persists for 5 s. These settings are fixed for every experimental cell.

3) Local correction and replay storage: The local residual in Eq. (10) remains in the same units as the base value. The count term in Eq. (9) is a decaying pessimism heuristic rather than a calibrated confidence bound. Replay labels, target shares, loss multipliers, precedence, and sparse-class fallback are specified in Section IV-D and Table I.

## REFERENCES

[1] G. K. Agrawal and S. S. Heragu, “A survey of automated material handling systems in 300-mm semiconductorfabs,” IEEE transactions on semiconductor manufacturing, vol. 19, no. 1, pp. 112–120, 2006.

[2] I. Hwang and Y. J. Jang, “Q (λ) learning-based dynamic route guidance algorithm for overhead hoist transport systems in semiconductor fabs,” International Journal of Production Research, vol. 58, no. 4, pp. 1199– 1221, 2020.

[3] B.-I. Kim, J. Shin, S. Jeong, and J. Koo, “Effective overhead hoist transport dispatching based on the hungarian algorithm for a large semiconductor fab,” International Journal of Production Research, vol. 47, no. 10, pp. 2823–2834, 2009.

[4] K. Bartlett, J. Lee, S. Ahmed, G. Nemhauser, J. Sokol, and B. Na, “Congestion-aware dynamic routing in automated material handling systems,” Computers & Industrial Engineering, vol. 70, pp. 176–182, 2014.

[5] S. Gupta, J. J. Hasenbein, and S. Park, “Improving scheduling and control of the ohtc controller in wafer fab amhs systems,” Simulation Modelling Practice and Theory, vol. 107, p. 102190, 2021.

[6] J. Boyan and M. Littman, “Packet routing in dynamically changing networks: A reinforcement learning approach,” Advances in neural information processing systems, vol. 6, 1993.

[7] H. Hasselt, “Double q-learning,” Advances in neural information processing systems, vol. 23, 2010.

[8] M. Zhang, L. Wang, F. Qiu, and X. Liu, “Dynamic scheduling for flexible job shop with insufficient transportation resources via graph neural network and deep reinforcement learning,” Computers & Industrial Engineering, vol. 186, p. 109718, 2023.

[9] F. Wang and J. Lin, “Performance evaluation of an automated material handling system for a wafer fab,” Robotics and Computer-Integrated Manufacturing, vol. 20, no. 2, pp. 91–100, 2004.

[10] J. Wan and H. Shin, “Predictive vehicle dispatching method for overhead hoist transport systems in semiconductor fabs,” International Journal of Production Research, vol. 60, no. 10, pp. 3063–3077, 2022.

[11] A. Benzoni, C. Yugma, and P. Bect, “Vehicle look-ahead dispatching for overhead hoist transport system in semiconductor manufacturing,” IEEE Transactions on Semiconductor Manufacturing, vol. 36, no. 1, pp. 130–138, 2023.

[12] K. Ahn and J. Park, “Cooperative zone-based rebalancing of idle overhead hoist transportations using multi-agent reinforcement learning with graph representation learning,” IISE Transactions, vol. 53, no. 10, pp. 1140–1156, 2021.

[13] C.-W. Chou, W.-C. Chiu, and Y.-T. Hsu, “Multiagent reinforcement learning-based dispatching model for overhead hoist transfer in automated material handling system,” Computers & Industrial Engineering, vol. 204, p. 111109, 2025.

[14] S. Lee, Y. Kim, H. Kahng, S.-K. Lee, S. Chung, T. Cheong, K. Shin, J. Park, and S. B. Kim, “Intelligent traffic control for autonomous vehicle systems based on machine learning,” Expert Systems with Applications, vol. 144, p. 113074, 2020.

[15] K. Ahn, K. Lee, J. Yeon, and J. Park, “Congestion-aware dynamic routing for an overhead hoist transporter system using a graph convolutional gated recurrent unit,” IISE Transactions, vol. 54, no. 8, pp. 803–816, 2022.

[16] J. Choi, T. Yu, and D. G. Choi, “Dynamic OHT routing using travel time approximation based on deep neural network,” IEEE Access, vol. 12, pp. 6900–6911, 2024.

[17] J. Lee and S. Lee, “Separable contextual graph neural networks to identify tailgating-oriented traffic congestion,” Expert Systems with Applications, vol. 254, p. 124354, 2024.

[18] Y. Kang, K. Im, S. Lee, and S. Cho, “Harmonyrouting with traffic impact prediction based on graph neural network for large-scale semiconductor fabrication,” Engineering Applications ofArtificial Intelligence, vol. 167, p. 113850, 2026.

[19] Y. Kang, S. Lyu, J. Kim, B. Park, and S. Cho, “Dynamic vehicle traffic control using deep reinforcement learning in automated material handling system,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 1, pp. 9949–9950, 2019.

[20] B. Koo, Y. Kim, J. Choi, S. Jun, and Y. Shin, “Alleviation of oht vehicle congestion in semiconductor fab with dynamic link weight control: A reinforcement learning approach,” International Journal of Production Research, pp. 1–26, 2026.

[21] J. Lee, J. Park, S. Hong, I. Hwang, S. Hwang, Y. J. Jang, D. Shin, and J. Lee, “Autonomous robot orchestration solution for oht with active q routing and digital twin,” IEEE Transactions on Semiconductor Manufacturing, 2025.

[22] S. P. M. Choi and D.-Y. Yeung, “Predictive Q-routing: A memory-based reinforcement learning approach to adaptive traffic control,” in Advances in Neural Information Processing Systems 8. MIT Press, 1996, pp. 945–951.

[23] S. Hong, I. Hwang, and Y. J. Jang, “Practical q-learning-based routeguidance and vehicle assignment for oht systems in semiconductor fabs,” IEEE Transactions on Semiconductor Manufacturing, vol. 35, no. 3, pp. 385–396, 2022.

[24] X. You, X. Li, Y. Xu, H. Feng, J. Zhao, and H. Yan, “Toward packet routing with fully distributed multiagent deep reinforcement learning,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 2, pp. 855–868, 2022.

[25] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015.

[26] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 30, no. 1, 2016.

[27] Q. Ao, Y. Zhou, W. Guo, W. Wang, and Y. Ye, “Dynamic path planning scheme for OHT in AMHS based on map information double deep Qnetwork,” Electronics, vol. 13, no. 22, p. 4385, 2024.

[28] T. Schaul, D. Horgan, K. Gregor, and D. Silver, “Universal value function approximators,” in Proceedings of the 32nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 37. PMLR, 2015, pp. 1312–1320.

[29] T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, I. Osband, G. Dulac-Arnold, J. P. Agapiou, J. Z. Leibo, and A. Gruslys, “Deep Q-learning from demonstrations,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 3223–3230.

[30] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative q-learning for offline reinforcement learning,” Advances in neural information processing systems, vol. 33, pp. 1179–1191, 2020.

[31] S. Fujimoto and S. S. Gu, “A minimalist approach to offline reinforcement learning,” Advances in neural information processing systems, vol. 34, pp. 20 132–20 145, 2021.

[32] S. Lee, Y. Seo, K. Lee, P. Abbeel, and J. Shin, “Offline-to-online reinforcement learning via balanced replay and pessimistic Q-ensemble,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 164. PMLR, 2022, pp. 1702–1712.

[33] H. Zhang, W. Xu, and H. Yu, “Policy expansion for bridging offline-toonline reinforcement learning,” arXiv preprint arXiv:2302.00935, 2023.

[34] M. Nakamoto, S. Zhai, A. Singh, M. S. Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine, “Cal-QL: Calibrated offline RL pre-training for efficient online fine-tuning,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 62 244–62 269.

[35] R. Kong, C. Wu, C.-X. Gao, Z. Zhang, and M. Li, “Efficient and stable offline-to-online reinforcement learning via continual policy revitalization.” in IJCAI, 2024, pp. 4317–4325.

[36] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” arXiv preprint arXiv:1511.05952, 2015.

[37] D. Isele and A. Cosgun, “Selective experience replay for lifelong learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 3302–3309.

[38] S. Bradtke and M. Duff, “Reinforcement learning methods for continuous-time markov decision problems,” Advances in neural information processing systems, vol. 7, 1994.

[39] A. Y. Ng, D. Harada, and S. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in Icml, vol. 99. Citeseer, 1999, pp. 278–287.