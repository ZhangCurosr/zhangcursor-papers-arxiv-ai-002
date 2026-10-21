# PULSE: Personalized Opportunistic Learning for Wearable Devices

Mattia G. Campana∗, Dariush Salami†, Hüseyin Yiğitler†, Stephan Sigg † ∗IIT-CNR, m.campana@iit.cnr.it

†Aalto University, {dariush.salami, huseyin.yigitler, stephan.sigg}@aalto.fi

Abstract—Wearable devices support personalized health and well-being applications, but post-deployment data can difer substantially from those used for training, while purely local adaptation may be limited by the data available on a single device during real-world operation. Conventional federated and decentralized learning rely on recurrent communication and stable connectivity, whereas existing opportunistic approaches often exploit available encounters without adequately considering their utility, compatibility, or communication cost.

We propose PULSE, an event-driven framework for personalized opportunistic learning that treats each encounter as an online collaboration decision, selecting either no interaction or a promising nearby peer. Peer gradients are incorporated only when compatible with the local learning direction, while past interactions guide future peer selection. Across human activity recognition, fall detection, and stress recognition under synthetic and real contact traces, PULSE preserves or improves personalized classification performance, while reducing peerassisted exchanges by 80–94% compared with opportunistic model averaging. On-device experiments further demonstrate its feasibility on wearable hardware: peer-assisted operations require limited energy, while the complete implementation occupies 149.8 KiB of RAM and 561.8 KiB of flash. The obtained results show that selective opportunistic collaboration can provide efective personalization while limiting communication and resource demands on constrained wearable devices.

Index Terms—Opportunistic Learning, Personalized Learning, Wearable Devices, Decentralized Learning, On-Device Learning

## I. Introduction

Wearable devices support applications for personal wellbeing and health, such as activity recognition, physiological monitoring, and afective-state assessment, by continuously collecting sensor data [1], [2]. They are often designed as sensing and inference endpoints, with machine learning models trained before deployment and subsequently executed on-device to provide low-latency predictions while reducing the need to continuously transmit raw observations [3]. Yet, the data observed after deployment may difer substantially from those used during training because of person-specific behaviours and physiological characteristics, sensing hardware and placement, and changes in daily routines or operating conditions [4], [5].

Although local adaptation can mitigate this issue, a single wearable may not collect suficient labeled data to learn an efective personalized model in isolation [6]. Collaborative learning ofers a complementary source of information: devices can improve their models using knowledge acquired by others without directly sharing raw observations [7], [8].

Human mobility naturally brings nearby devices into contact, creating intermittent opportunities to exchange learning information and to support collaborative adaptation. This setting has motivated Opportunistic Learning (OL), in which decentralized model adaptation is driven by transient device-to-device encounters rather than scheduled communication rounds or persistent network connections [9]. Existing OL approaches have explored gradient exchange, contact-aware collaboration policies, and the transfer of model components among encountered devices [10]. While they demonstrate that intermittent encounters can support collaborative learning without centralized orchestration, the heterogeneity of wearable data makes indiscriminate knowledge transfer particularly problematic: a peer update may be irrelevant or harmful when diferent sensing conditions, activity distributions, or locally adapted representations are merged.

In this work we introduce PULSE, a personalized opportunistic learning mechanism that makes peer collaboration explicitly selective at two levels. First, at each encounter, a lightweight utility-guided policy selects one promising peer or chooses no contact when communication is not expected to be useful. Second, rather than averaging models or accepting a peer update unconditionally, PULSE evaluates the agreement between the remote and local gradients and incorporates the remote signal only when it is positively aligned with the local learning direction. Each device fine-tunes its full model on private data, while peer interactions exchange only the current prediction head and its corresponding gradient, keeping the collaboration lightweight and preserving raw-data locality.

The contributions of this work are

• PULSE, an event-driven framework for Personalized OL that combines local adaptation with peer-assisted updates;

• Selective collaboration through a two-level mechanism that determines whether a peer should be contacted and whether its returned gradient should influence the local training direction;

• Comprehensive simulation results across three heterogeneous wearable applications and synthetic and real contact traces, considering both classification performance and communication demand;

• a Prototype implementation to characterize the latency, memory, and energy requirements of real wearables;

• an Open Codebase, including the full implementation and experimental code, publicly available at https://anonymous.4open.science/r/PULSE-F68A.

## II. Related Work

Federated Learning enables multiple devices to improve a learning model without directly sharing their raw data, typically by periodically communicating local model updates to a coordinating server [11]. This paradigm can reduce the exposure of personal sensing data and has motivated extensive work on mobile and edge intelligence [8]. In many contemporary edge deployments, pervasive connectivity and access to a coordinating infrastructure may indeed be available. However, conventional Federated Learning typically relies on frequent communication rounds and on the transfer of model parameters or gradients, which can impose substantial communication, energy, and computation overhead. Such requirements may be poorly aligned with pervasive sensing scenarios involving resource-constrained mobile and wearable devices, for which continuous participation in collaborative learning may be impractical or ineficient.

Decentralized Federated Learning removes the central coordinator and enables devices to exchange learning updates directly with peers [12]. This improves robustness to infrastructure failures and avoids reliance on a single aggregation point. However, most decentralized optimization methods assume frequent peer-to-peer exchanges over persistent or repeatedly available communication graphs, typically through iterative gossip protocols. While these assumptions may be reasonable for smartphones or single-board computers connected through Wi-Fi or local networks, repeated computation and communication can impose substantial energy costs on resource-constrained wearable devices [13].

A line of work closer to our research considers collaboration driven by physical encounters among mobile devices. Lee et al. introduced Opportunistic Federated Learning, in which devices personalize locally trained models by incorporating knowledge from encountered peers and use mobility and contact prediction to decide whether collaboration is worthwhile [9]. This formulation importantly recognizes that encounters difer in their expected utility and duration, rather than treating communication opportunities as uniformly available. At the same time, the approach is centered on exchanging learning updates (i.e., gradients) between devices and relies on estimating future contact properties, which may be inaccurate or unavailable in highly dynamic environments.

More recently, OPTIMIST considers opportunistic federated transfer learning under intermittent connectivity and without centralized orchestration during the learning process [10]. It combines a frozen, pre-trained backbone with a trainable classifier head that is partitioned into nonoverlapping subnetworks. Each subnetwork is assigned to a node, trained locally, and transferred to an encountered peer, thereby enabling its sequential exposure to multiple local datasets while avoiding parameter conflicts among parallel updates. This design substantially reduces the computational burden relative to end-to-end collaborative training and is evaluated through both simulations and an Android-based prototype.

However, the framework presupposes an initialization procedure that creates the partition of the classifier head and assigns a distinct subnetwork to each participating node. Although this procedure need not entail centralized coordination during subsequent encounters, it implicitly requires either a bootstrap component with global knowledge of the participating nodes or a shared deterministic assignment mechanism; the paper does not further specify how this initialization would be realized in a fully autonomous deployment. Moreover, OPTIMIST formulates learning as the optimization of a classifier head over the virtually merged data of all nodes. Its primary objective is therefore collaborative generalization: subnetworks circulate to incorporate information from multiple datasets, rather than being selected or adapted according to the specific needs, data distribution, or compatibility of an individual user. While nodes may hold diferent local head instances at a given time because of their distinct encounter histories, this divergence is an efect of asynchronous exchanges rather than an explicit personalization mechanism. Finally, its mobile evaluation targets Android smartphones and parameter-level subnetwork transfer, whose resource requirements remain diferent from those of microcontroller-class wearable devices.

## A. Real-device evaluation of collaborative edge learning

Experimental validation on physical devices is necessary to complement simulation-based analyses, which cannot fully capture the execution time, memory footprint, wireless latency, and energy consumption associated with collaborative learning. Prior work has developed Federated Learning testbeds on resource-constrained edge platforms, ofering valuable insights into the discrepancy between abstract communication rounds and their practical realization [14], [15]. These studies enable an end-to-end assessment of system-level behavior beyond predictive performance alone. However, the considered platforms generally rely on single-board computers, smartphones, or other comparatively capable edge devices, typically operating over Wi-Fi or an accessible local network.

On the other hand, wearable sensing introduces a distinct operating regime: devices have tighter energy and memory budgets, may collect highly personal and userdependent data, and encounter peers through short-lived, irregular contacts [16]. Consequently, realistic assessment should jointly account for learning quality, heterogeneous sensing data, contact dynamics, transmitted payload, execution latency, memory consumption, and energy expenditure. This motivates evaluations that combine controlled simulations, including real-world mobility patterns or contact traces, with measurements on actual resourceconstrained wearable hardware.

Algorithm 1 PULSE at node i   
Require: Local model $f _ { i } = h _ { i } \circ { \mathcal { E } } _ { i } .$ , local data $\mathcal { D } _ { i } .$ , local   
training steps ${ \mathcal { R } } ,$ peer-utility estimates $\mathcal { U } _ { i }$   
1: Upon OnLocalTraining:   
2: for $r = 1 , \ldots , \mathcal { R }$ do   
3: f<sup>r+1</sup> Update $f _ { i } ^ { r }$ with Eq. 1   
4: end for   
5: Upon OnPeerDiscovery $\cdot ( \mathcal { N } _ { i } ( t ) )$   
6: j  SelectPeer $\cdot ( \mathcal { N } _ { i } ( t ) , \dot { \mathcal { U } } _ { i } )$   
7: ${ \cal B } _ { i } \gets$ Sample a minibatch from $\mathcal { D } _ { i }$   
8: $g _ { i } \gets \nabla _ { h } \ell _ { i } ( h _ { i } \circ \mathcal { E } _ { i } ; B _ { i } )$   
9: if $j = \emptyset$ then   
10: $h _ { i }  h _ { i } - \eta g _ { i }$   
11: else   
12: send a copy of $h _ { i }$ to $j$   
13: $g _ { j } \gets$ receive remote gradient from $j$   
14: $a _ { i j } $ Gradient agreement with Eq. 4   
15: $\alpha _ { i j }$ Peer contribution with Eq. 6   
16: $h _ { i } \gets$ Update $h _ { i }$ with Eq. 7   
17: $\mathcal { U } _ { i } [ j ] $ Update utility with Eq. 8   
18: end if

## III. The PULSE algorithm

PULSE is an event-driven mechanism specifically designed for personalized learning on wearable and edge devices experiencing brief and sporadic peer encounters. Before deployment, a shared encoder is pretrained on representative data in a centralized bootstrap phase. Each node i then initializes its local model as $f _ { i } = h _ { i } \circ \mathcal { E } _ { i }$ , where $\mathcal { E } _ { i } : \mathcal { X }  \mathbb { R } ^ { d }$ is initialized from the pretrained encoder and maps an input observation to a d-dimensional latent representation, while a lightweight multilayer-perceptron head $h _ { i } : \mathbb { R } ^ { d }  \mathcal { V }$ maps this representation to the output space.

After deployment, each node locally fine-tunes $f _ { i }$ on its private, potentially evolving data distribution, allowing the model to adapt to user-specific behaviours and sensing conditions while retaining the task-aware initialization obtained during bootstrap. Rather than averaging complete models or exchanging raw observations, PULSE restricts peer-assisted interactions to gradients of $h _ { i }$ , thus keeping both the communication payload and peer-side computation lightweight.

Since information received from an arbitrary peer may be irrelevant or harmful, PULSE selectively incorporates remote learning signals. During an encounter, node i sends its current $h _ { i }$ to a selected peer $j ,$ which evaluates a temporary copy using its local $\mathcal { E } _ { j }$ and a private minibatch; node i then compares the resulting gradient with its local learning direction and incorporates it only when their cosine agreement is positive.

Algorithm 1 summarizes the operations performed by node i upon local-training and peer-discovery events.

## A. Local Update

At scheduled times, node i fine-tunes $f _ { i }$ using its local dataset $\mathcal { D } _ { i }$ , thus supporting continuous adaptation even when useful peer encounters are unavailable. At each local update event, node i performs R stochastic-gradient steps, as detailed in lines 2–5 of Algorithm 1:

$$
f _ { i } ^ { r + 1 } = f _ { i } ^ { r } - \eta \nabla _ { f } \ell _ { i } ( f _ { i } ^ { r } ; B _ { i } ^ { r } ) ,\tag{1}
$$

where η is the learning rate, $B _ { i } ^ { r }$ is a minibatch sampled from $\mathcal { D } _ { i }$ , and $\ell _ { i }$ denotes the local prediction loss, e.g., cross-entropy for classification.

## B. Opportunistic Peer-Assisted Update

Whenever nearby devices are detected, node i forms the set of reachable peers at time $t , \ N _ { i } ( t )$ , and selects one peer $j \in \mathcal { N } _ { i } ( t )$ according to the utility-guided policy detailed in Section III-C. The selected peer provides i with an additional gradient-based learning signal: information about how its current $h _ { i }$ should change to better fit data observed by peer j.

Specifically, node i sends its current head $h _ { i }$ to peer $j$ (line 10). Peer $j$ treats the received parameters as a temporary copy, leaving its own personalized head unchanged. Using a local minibatch $B _ { j } { \mathrm { : } }$ , it computes the gradient of its loss with respect to the received head (line 11). This gradient identifies the parameter direction that would reduce the loss of $h _ { i }$ on peer $j ^ { \prime } { : }$ s observations, and is therefore returned to node i as the learning signal associated with that encounter. This produces the remote gradient:

$$
g _ { j } = \nabla _ { h } \ell _ { j } ( h ; \mathcal { E } _ { j } , B _ { j } ) ,\tag{2}
$$

where $\ell _ { j } ( \cdot )$ is the loss of $j$ and $B _ { j }$ is a minibatch drawn from $D _ { j }$ . In parallel, node i computes its local gradient at the same parameter point (line 12),

$$
g _ { i } = \nabla _ { h } \ell _ { i } ( h ; \mathcal { E } _ { i } , B _ { i } ) ,\tag{3}
$$

using a local batch of data $B _ { i }$

Gradient-based messages are a standard primitive in distributed optimization and collaborative learning [17]. Here, however, the remote gradient has a specific role: since $g _ { j }$ is evaluated at $h _ { i }$ , it describes how peer $j ^ { \circ } \mathrm { s }$ data would modify node i’s current personalized head, rather than how it would modify peer $j ^ { ; }$ ’s own model. PULSE can therefore assess whether this candidate gradient direction is compatible with node i’s local learning objective before applying it.

To this end, we measure the cosine agreement between the two gradients (line 13):

$$
a _ { i j } = \frac { \langle g _ { i } , g _ { j } \rangle } { \operatorname* { m a x } ( \| g _ { i } \| \| g _ { j } \| , \epsilon ) } ,\tag{4}
$$

where $\epsilon > 0$ prevents numerical instability when either gradient has negligible norm. The resulting score is approximately bounded in [ 1, 1] and measures directional agreement independently of gradient magnitude. A positive $a _ { i j }$ indicates that the remote gradient $g _ { j }$ has a component aligned with i’s local descent direction and is therefore expected to support the reduction of node i’s loss. Instead, a negative value identifies a conflicting direction, while $a _ { i j } = 0$ indicates that there is no benefit from the remote update.

This rationale is related to gradient-conflict mitigation methods in multi-task learning, which identifies incompatible gradients before combining them [18]. In PULSE, we use a lightweight alternative suited to brief encounters: rather than projecting or jointly optimizing conflicting gradients, it simply discards remote signals that do not agree with the local objective. Moreover, even when $g _ { j }$ is aligned with $g _ { i } ,$ its norm may difer substantially because of local loss scales, batch composition, or data heterogeneity. Without additional control, a large remote gradient could dominate the update despite providing only one peer’s evidence. We therefore rescale its magnitude to match that of the local gradient:

$$
\widetilde { g } _ { j } = \sqrt { \frac { \lVert g _ { i } \rVert ^ { 2 } } { \operatorname* { m a x } ( \lVert g _ { j } \rVert ^ { 2 } , \epsilon ) } } g _ { j } .\tag{5}
$$

The floor is therefore applied to the squared remote norm before taking the square root, matching the executable implementation. When $\| g _ { j } \| ^ { 2 } > \epsilon .$ , the resulting vector preserves the remote direction while matching its norm to that of $g _ { i }$ . This provides a simple safeguard against magnitude mismatches between local and remote updates. It is related in spirit to gradient-normalization approaches that account for learning signals with diferent magnitudes [19], although PULSE does not balance multiple task losses and uses normalization solely to scale the accepted remote gradient.

The node i also maintains a signed utility estimate $\mathcal { U } _ { i } [ j ] \in [ - 1 , 1 ]$ for each peer $j .$ This value summarizes whether, in previous encounters, the peer has tended to provide gradients aligned with or conflicting with node $i \mathrm { \ ' } _ { \mathrm { S } }$ local objective. The contribution of the remote signal (line 14) is set to

$$
\alpha _ { i j } = \bar { \alpha } \operatorname* { m a x } ( \mathcal { U } _ { i } [ j ] , 0 ) \operatorname* { m a x } ( a _ { i j } , 0 ) ,\tag{6}
$$

where $\bar { \alpha } \in ( 0 , 1 )$ is a fixed upper bound on the remote contribution. Since $\alpha _ { i j } \leq \bar { \alpha } < 1$ , the local gradient always retains a non-zero weight in the update. Thus, a peer can contribute substantially only when it has been useful in previous encounters and its current gradient is strongly aligned with the local gradient. Negative historical utility never increases the remote contribution; however, it remains relevant to the peer-selection policy, which may decide not to initiate communication with a peer that has repeatedly produced conflicting signals.

The local head of node i is then updated (line 15) as

$$
h _ { i }  h _ { i } - \eta [ ( 1 - \alpha _ { i j } ) g _ { i } + \alpha _ { i j } \widetilde { g } _ { j } ] .\tag{7}
$$

The update combines the local learning direction with a compatible, norm-matched remote direction. Since $\alpha _ { i j } \leq$ ${ \bar { \alpha } } ,$ choosing $\bar { \alpha } < 1$ ensures that the local gradient always retains a non-zero contribution to the update. When $a _ { i j } >$ 0, the peer provides a direction consistent with the change suggested by node i’s own data. Otherwise, $\alpha _ { i j } = 0$ and (7) reduces to a standard local gradient step.

After a peer-assisted update, node i updates the utility of the selected peer (line 16) using the observed gradient agreement:

$$
\mathcal { U } _ { i } [ j ]  ( 1 - \mu ) \mathcal { U } _ { i } [ j ] + \mu a _ { i j } ,\tag{8}
$$

where $\mu \in ( 0 , 1 ]$ controls the weight assigned to the most recent interaction. This recursion is an exponential moving average of the signed gradient agreement: positive values indicate that peer j has generally supplied compatible learning directions, whereas negative values indicate recurrent conflict with node i’s local objective. The estimate therefore retains information from past encounters while adapting to changes in the peer’s usefulness over time.

## C. Peer Selection Policy

At each peer-discovery event, node i must decide whether communication is worthwhile and, if so, which currently reachable peer should be contacted. This is a sequential decision problem: the usefulness of a peer can only be assessed after interacting with it, while its utility may vary over time as the users’ contexts and data distributions evolve.

PULSE addresses this decision through a lightweight policy inspired by the Upper Confidence Bound (UCB) principle [20]. Accordingly, each reachable peer is treated as an alternative whose utility is learned online from the gradient agreements observed in previous encounters. Crucially, PULSE also includes a no-contact action, denoted by ∅, enabling node i to skip communication when none of the available peers appears suficiently promising, rather than being forced to contact an arbitrary device (line 9). Selecting ∅ means that node i does not engage in peer collaboration, while still taking advantage of the update opportunity through the local instance of Eq. (7), with mixing coeficient equal to zero. This avoids unnecessary communication without interrupting local adaptation.

For every discovered peer j, node i maintains a utility estimate $\mathcal { U } _ { i } [ j ]$ and a selection count $n _ { i j }$ . The former summarizes whether the peer has historically provided aligned or conflicting gradients, whereas the latter measures the amount of evidence collected about that peer. Let $s _ { i }$ denote the number of peer-selection opportunities observed so far by node i. At a peer-discovery event, node i selects

$$
j _ { i } ^ { \star }  \operatorname * { a r g m a x } _ { j \in \mathcal { N } _ { i } ( t ) \cup \{ \emptyset \} } \{ \begin{array} { l l } { \mathcal { U } _ { i } [ j ] + \beta \sqrt { \frac { \log ( 2 + s _ { i } ) } { 1 + n _ { i j } } } , } & { j \neq \emptyset , } \\ { 0 , } & { j = \emptyset , } \end{array}\tag{9}
$$

where $\beta > 0$ controls the degree of exploration. The no-contact action has index zero, providing the reference against which the estimated benefit of a peer exchange is evaluated.

For a reachable peer j, the first term favors devices that have previously supplied gradients compatible with node i’s local objective. The second term is an exploration bonus: it is larger for peers selected only a few times and decreases as $n _ { i j }$ grows. Consequently, a newly discovered peer is initially explored, whereas a peer that has repeatedly produced conflicting gradients may receive an index below that of the no-contact action. In this case, PULSE avoids communication even if that peer is the only currently reachable device. Moreover, the policy uses the encounter index $s _ { i } .$ , rather than physical time, so that its exploration behavior evolves only when node i has an actual opportunity to make a communication decision.

After the decision, $s _ { i }$ is incremented. If $j _ { i } ^ { \star } \neq \emptyset ,$ , node i performs the peer-assisted update and increments $n _ { i j _ { i } ^ { \star } }$ after observing the resulting gradient agreement. Over time, the policy identifies reachable peers that tend to provide beneficial learning signals, while avoiding exchanges whose expected contribution does not justify their cost.

## IV. Simulation Results

We firstly evaluate PULSE in a controlled eventdriven simulator of decentralized learning among wearable devices under intermittent device-to-device connectivity. Each node represents one user and retains only its associated sensor data, while the simulator replays local-training opportunities and peer contacts over time.

We consider heterogeneous wearable applications and synthetic and real-world contact traces, which vary in sensing modality, learning task, user-data heterogeneity, and encounter dynamics. We compare PULSE with representative opportunistic-learning baselines in terms of classification performance and communication demand, measured by the number of completed peer-assisted message exchanges.

## A. Simulated Scenarios

We consider three wearable sensing applications: Human Activity Recognition (HAR), Fall Detection (FD), and Stress Recognition (SR). As summarized in Table I, the scenarios difer in their sensing modalities, number of users, label spaces, and class imbalance. Human activity recognition (HAR). We adopt a cross-dataset setup based on UCI-HAR [21] and HHAR [22]. UCI-HAR comprises smartphone inertial recordings from 30 users performing six daily activities and is used solely to obtain a common pretrained initialization. Decentralized simulations are conducted on HHAR, which provides 87.6K accelerometer windows from nine users collected with heterogeneous smartphones and smartwatches. This setting captures both user- and device-dependent distribution shifts between pretraining and deployment. For both datasets, we use three-axis accelerometer windows of 128 samples, independently normalized through per-axis z-score standardization. Fall detection (FD). SisFall [23] provides wearable accelerometer and gyroscope recordings from 38 users performing activities of daily living and falls. We formulate the task as binary fall versus non-fall classification. As shown in Table I, its majority class accounts for 67.4% of the samples, making it the most imbalanced scenario. Stress recognition (SR). We use WESAD [24], which contains synchronized wrist- and chest-worn sensor measurements from 15 users exposed to diferent afective conditions. We consider its three-class stress-recognition task. Unlike the inertial scenarios, its heterogeneity mainly stems from subject-specific physiological responses.

TABLE I: Summary of the evaluation datasets.
<table><tr><td>Dataset</td><td>Scenario</td><td>Users</td><td>Samples</td><td>Labels (Max. %)</td></tr><tr><td>UCI-HAR</td><td>HAR</td><td>30</td><td>10.32 K</td><td>6 (18.9)</td></tr><tr><td>HHAR</td><td>HAR</td><td>9</td><td>87.6 K</td><td>6 (19.5)</td></tr><tr><td>SisFall</td><td>FD</td><td>38</td><td>55.1 K</td><td>2 (67.4)</td></tr><tr><td>WESAD</td><td>SR</td><td>15</td><td>1 K</td><td>3 (54.5)</td></tr></table>

For FD and SR, we partition the available subjects into disjoint pretraining, validation, and simulation sets. Pretraining data are used to learn a task-specific initial encoder, validation data are used exclusively for checkpoint selection, and simulation data remain unseen until the controlled experiments. In FD, we use 22 subjects (29,187 samples, 53.0%) for pretraining, 6 subjects (9,832 samples, 17.8%) for validation, and 10 subjects (16,070 samples, 29.2%) for the decentralized simulation. The corresponding SR partitions contain 3 subjects and 204 samples (19.9%), 2 subjects and 140 samples (13.6%), and 10 subjects and 682 samples (66.5%), respectively.

HAR follows a diferent cross-dataset protocol. UCI-HAR contributes 26 subjects and 8,921 samples (86.6%) to pretraining, and four subjects and 1,378 samples (13.4%) to validation. All nine HHAR users and their 87,634 samples are instead reserved for the decentralized simulation. Consequently, the HAR setting evaluates adaptation not only to previously unseen users, but also to a distinct target dataset collected with heterogeneous smartphones and smartwatches.

![](images/93434e322ddd4f4867e8421753b2ce89c1c3111de733ffdb7972a43e6dfe8b10.jpg)  
(a) Training loss.

![](images/88bd3fcb6d65dd1135ef18df87bb5e5762f221333b6c038f9503f96629dc3656.jpg)  
(b) Validation F1-score.  
Fig. 1: Pretraining curves for the task-specific models.

![](images/9cd56814bf049be9495f19d24745826df32bb910c05018421dde20ffe5669cf1.jpg)  
Fig. 2: Temporal evolution of contact opportunities.

## B. Models Pretraining

To provide all nodes with a common yet task-aware initialization, we pretrain a lightweight encoder for each scenario before the decentralized simulations. The architectures are tailored to the available sensing modalities while maintaining a common 64-dimensional embedding space and a limited computational footprint. During pretraining, each encoder is paired with an auxiliary MLP classifier comprising one 64-unit hidden layer. The selected encoder checkpoint then initializes every simulated node, which locally fine-tunes it together with its own personalized classifier.

For HAR, we use a compact one-dimensional CNN with two temporal convolutional blocks over the three-axis accelerometer signal. FD uses a six-channel temporal convolutional network with residual dilated blocks to process accelerometer and gyroscope windows. For SR, separate convolutional branches encode the accelerometer, bloodvolume pulse, electrodermal activity, and temperature signals; their representations are fused before classification. All models are optimized with AdamW and cross-entropy loss using a learning rate of 10<sup>−3</sup>. We monitor validation F1-score, retaining the checkpoint with the highest value, and restore it before deployment. Early stopping uses a minimum improvement of 0.001: HAR and FD stop after 20 and 5 epochs without improvement, respectively, whereas SR is trained for the maximum budget of 100 epochs with a patience of 50.

Figure 1 shows the pre-training dynamics, with the horizontal axis normalized by the training budget of each scenario. The training loss decreases consistently in all cases. The selected HAR, FD, and SR encoders achieve validation F1-scores of 0.836, 0.783, and 0.746 at epochs 8, 18, and 53, respectively. Validation F1 is not strictly monotonic, particularly for FD and SR, as expected under stochastic optimization and limited validation data. Selecting the best validation checkpoint therefore provides a robust initialization before local personalization in the opportunistic scenarios.

## C. Contact Traces

We replay time-stamped pairwise contact traces to determine when devices can exchange learning signals. A contact event indicates that the corresponding node pair can communicate at a given time. Repeated observations of the same undirected pair within a snapshot are removed. We consider two controlled synthetic mobility models and the real-world Cambridge/Haggle Bluetooth trace [25].

1) Synthetic traces: We generate traces using the Community [26] and Edge-Markov [27] models. Each synthetic trace contains ten nodes, spans 24 hours, and is sampled every 20 seconds. For each model, we generate three independent realizations using seeds 7, 42, and 123. For both models, we consider a Dense and a Sparse connectivity regime that difer in the frequency of communication opportunities while preserving the defining property of the respective mobility model. Dense traces provide approximately 4.5–4.9 active contacts per snapshot, corresponding to a mean occupancy of about 10% of all possible node pairs. Sparse traces provide approximately 0.8–1.0 contacts per snapshot, or about 2% occupancy. They therefore reduce the rate of peer encounters by roughly a factor of five, creating a more constrained setting for opportunistic learning.

In Community traces, contacts remain predominantly local: approximately 86.6% and 86.3% of events occur within communities in the Dense and Sparse regimes, respectively. For each mobility model, the two regimes retain the same node population and contact-generation mechanism, while varying only the overall frequency of contact opportunities. The Dense–Sparse comparison therefore isolates the efect of peer availability under the same underlying mobility model.

2) Real trace: We additionally use the Cambridge/Haggle trace, which records Bluetooth encounters among iMotes carried by members of a research group at the University of Cambridge [25]. It contains 12 nodes, spans approximately 126.6 hours, and is sampled every 120 seconds. Its mean pair occupancy is 4.14%, intermediate between the controlled Dense and Sparse regimes. Unlike the approximately stationary synthetic traces, however,

Cambridge exhibits temporally uneven activity, with contacts concentrated in a limited number of intervals. Figure 2 highlights both the controlled separation between Dense and Sparse conditions and the irregular temporal dynamics of the real trace. Together, these traces assess the methods under frequent, scarce, and naturally bursty peer-learning opportunities.

## D. Baselines

We compare PULSE with four baselines. Local trains exclusively on private data, without any communications. OppAVG opportunistically averages the models of all reachable peers. OppFL sends the local model to a peer, receives a gradient computed on the peer’s private minibatch, and aggregates it with the local gradient. OPTIMIST partitions the classifier into N subnetworks, one initially assigned to each of the N nodes; mutually matched nodes exchange their active subnetworks and update only the received one. It performs no update without an accepted exchange.

All methods start from the same pretrained model, update both the encoder and prediction head, and never exchange raw data. They use two SGD steps, batch size of 16, and learning rate of 0.01. OppFL uses a zero similarity threshold, unit peer-gradient weight, two interaction rounds, and momentum aggregation, thus allowing the method to exploit all non-conflicting peer updates while preserving its original aggregation mechanism. For OPTI-MIST, we set the number of subnetworks as to the number of nodes in each simulation. For PULSE, we set the maximum mixing coeficient to $\bar { \alpha } = 0 . 5$ , the utility-update rate to $\mu = 0 . 2$ , and the UCB exploration coeficient to $\beta = 0 . 2 5$ . These values keep peer contributions bounded, let utility estimates react to recent interactions without becoming overly volatile, and provide limited exploration of previously untested peers.

## E. Classification Results

Table II reports the final performance across the three real-world sensing applications and five contact-trace configurations. For each simulation user, local data are split into 70% for training and 30% for testing, with Accuracy and Macro-F1 evaluated exclusively on the local test set. Local learning is already a strong baseline in several settings: each node is therefore trained and evaluated on data from the same user, all nodes start from a pretrained encoder, and the simulated users generally retain examples from every target class. Hence, collaboration is beneficial only when encountered peers provide complementary and compatible learning signals.

This efect is particularly evident in FD. Local, OppAVG, and PULSE reach the same performance with the Cambridge trace (0.92 Accuracy and 0.82 F1), and remain closely aligned in the dense synthetic scenarios (approximately 0.91 Accuracy and 0.81 F1). Sparse contacts lead only to a limited reduction for PULSE, whose F1 reaches

0.80 under Community–Sparse. Since FD is a binary task and its encoder is pretrained on data from other subjects of the same dataset, the strong Local results suggest limited headroom for collaboration. PULSE nevertheless remains competitive, showing that its peer-assisted updates do not introduce substantial negative transfer when local personalization is already efective.

A similar pattern emerges in SR. Local obtains the best Cambridge accuracy (0.98), while PULSE remains close at 0.97 accuracy and 0.91 F1. Under all synthetic traces, however, PULSE achieves the highest F1, reaching 0.88 in the dense configurations and 0.87–0.88 under sparse connectivity, while maintaining 0.95 accuracy throughout. Although the margins over Local are moderate, these results show that PULSE can exploit cross-user information without compromising personalized performance.

The benefits of collaboration are substantially clearer in HAR, where the encoder is pretrained on UCI-HAR and then adapted to users and devices from a diferent dataset. In the dense scenarios, PULSE reaches 0.82 accuracy and 0.81–0.82 F1, compared with 0.68 accuracy and 0.64 F1 for Local. OPTIMIST is the strongest baseline in these settings, but remains below PULSE (0.70–0.71 accuracy and 0.69–0.70 F1). PULSE also retains a clear advantage under sparse contacts, obtaining 0.76–0.77 accuracy and 0.75–0.76 F1. In contrast, OPTIMIST and OppFL fall to approximately 0.65/0.63 and 0.54/0.51, respectively, reflecting their stronger dependence on frequent successful encounters. With Cambridge, PULSE matches Local in accuracy (0.87) and slightly improves F1 (0.87 versus 0.86), whereas OppFL and OPTIMIST reach 0.55/0.52 and 0.59/0.57, respectively. This is consistent with the intermittent nature of the real trace, where long periods with few contacts penalize approaches whose optimization is primarily encounter-driven.

Overall, PULSE provides the most consistent behavior across the evaluation. It achieves the best accuracy and F1 in all HAR configurations, the highest Macro-F1 in every synthetic SR scenario, and remains comparable to Local on FD. The final-loss values follow the same trend, with PULSE attaining the lowest rounded loss in all HAR settings and in most synthetic FD and SR settings. These results indicate that selective collaboration can preserve strong local personalization when communication is unhelpful, while providing substantial gains when local data and pretrained representations are less well aligned with the target setting.

## F. Communication Eficiency

Table III reports the average number ( 1000) of peerassisted updates completed throughout the simulated scenarios. As expected, dense synthetic traces generate substantially more exploitable interactions than their sparse counterparts, while the Cambridge trace typically lies between the two regimes. OppAVG performs the largest number of updates, ranging from 5,788 to 41,801 across the considered settings. OppFL exhibits a similar communication demand in HAR and SR, whereas in FD it completes between 5,817 and 33,301 updates due to its similarity-based peer filtering. OPTIMIST reduces the number of interactions relative to these methods, but still performs between 5,044 and 21,358 updates.

TABLE II: Final test performance. Bold and underlined values indicate the best and second-best available methods, respectively, for each dataset, contact-trace configuration, and metric.
<table><tr><td></td><td></td><td colspan="3">Cambridge</td><td colspan="3">Community-Dense</td><td colspan="3">Community-Sparse</td><td colspan="3">Edge-Markov-Dense</td><td colspan="3">Edge-Markov-Sparse</td></tr><tr><td>Scenario</td><td>Method</td><td>Loss</td><td>Acc.</td><td>F1</td><td>Loss</td><td>Acc.</td><td>F1</td><td>Loss</td><td>Acc.</td><td>F1</td><td>Loss</td><td>Acc.</td><td>F1</td><td>Loss</td><td>Acc.</td><td>F1</td></tr><tr><td rowspan="5">HAR</td><td>Local</td><td>.40</td><td>.87</td><td>.86</td><td>.88</td><td>.68</td><td>.64</td><td>.89</td><td>.68</td><td>.64</td><td>.87</td><td>.68</td><td>.64</td><td>.88</td><td>.68</td><td>.64</td></tr><tr><td>OppAVG</td><td>.41</td><td>.86</td><td>.86</td><td>.96</td><td>.61</td><td>.56</td><td>.97</td><td>.61</td><td>.56</td><td>.96</td><td>.61</td><td>.55</td><td>.97</td><td>.61</td><td>.55</td></tr><tr><td>OppFL</td><td>1.05</td><td>.55</td><td>.52</td><td>.91</td><td>.65</td><td>.64</td><td>1.24</td><td>.54</td><td>.51</td><td>.94</td><td>.65</td><td>.64</td><td>1.20</td><td>.54</td><td>.51</td></tr><tr><td>OPTIMIST</td><td>.91</td><td>.59</td><td>.57</td><td>.86</td><td>.70</td><td>.69</td><td>.95</td><td>.65</td><td>.63</td><td>.83</td><td>71</td><td>.70</td><td>1.00</td><td>.65</td><td>.63</td></tr><tr><td>PULSE</td><td>.39</td><td>.87</td><td>.87</td><td>.55</td><td>.82</td><td>.82</td><td>.71</td><td>.77</td><td>.76</td><td>.58</td><td>.82</td><td>.81</td><td>.69</td><td>.76</td><td>.75</td></tr><tr><td rowspan="5">FD</td><td>Local</td><td>.09</td><td>.92</td><td>.82</td><td>19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td></tr><tr><td>OppAVG</td><td>.09</td><td>.92</td><td>.82</td><td>.19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td><td>.19</td><td>.91</td><td>.81</td></tr><tr><td>OppFL</td><td>.20</td><td>.85</td><td>.74</td><td>.30</td><td>.86</td><td>.76</td><td>.35</td><td>.84</td><td>.74</td><td>.29</td><td>.86</td><td>.76</td><td>.28</td><td>.84</td><td>.74</td></tr><tr><td>OPTIMIST</td><td>.31</td><td>.84</td><td>.74</td><td>.41</td><td>.84</td><td>.75</td><td>.40</td><td>.84</td><td>.74</td><td>.39</td><td>.85</td><td>.75</td><td>.36</td><td>.85</td><td>.75</td></tr><tr><td>PULSE</td><td>.08</td><td>.92</td><td>.82</td><td>.18</td><td>.91</td><td>.81</td><td>.21</td><td>.91</td><td>.80</td><td>.18</td><td>.91</td><td>.81</td><td>.20</td><td>.91</td><td>.81</td></tr><tr><td rowspan="5">SR</td><td>Local</td><td>.00</td><td>.98</td><td>.92</td><td>.10</td><td>.95</td><td>.87</td><td>.11</td><td>.94</td><td>.86</td><td>.10</td><td>.95</td><td>.87</td><td>.10</td><td>.95</td><td>.87</td></tr><tr><td>OppAVG</td><td>.00</td><td>.97</td><td>.92</td><td>.09</td><td>.93</td><td>.85</td><td>10</td><td>.93</td><td>.85</td><td>.09</td><td>.93</td><td>.84</td><td>.09</td><td>.94</td><td>.85</td></tr><tr><td>OppFL</td><td>.29</td><td>.84</td><td>.69</td><td>.21</td><td>.86</td><td>.75</td><td>.45</td><td>.83</td><td>.66</td><td>.25</td><td>.87</td><td>.75</td><td>.45</td><td>.83</td><td>.66</td></tr><tr><td>OPTIMIST</td><td>.16</td><td>.86</td><td>.77</td><td>.20</td><td>.88</td><td>.77</td><td>.34</td><td>.86 .95</td><td>.76 .87</td><td>.18 .04</td><td>.88 .95</td><td>.77</td><td>.29</td><td>.86</td><td>.75</td></tr><tr><td>PULSE</td><td>.01</td><td>.97</td><td>.91</td><td>.04</td><td>.95</td><td>.88</td><td>.06</td><td></td><td></td><td></td><td></td><td>.88</td><td>.05</td><td>.95</td><td>.88</td></tr></table>

TABLE III: Average number ( 1000) of completed peerassisted updates across the simulation.
<table><tr><td>Scenario</td><td>OppAVG</td><td>OppFL</td><td>OPTIMIST</td><td>PULSE</td></tr><tr><td>HAR-C</td><td>10.99</td><td>10.99</td><td>5.31</td><td>1.42</td></tr><tr><td>HAR-CD</td><td>33.35</td><td>33.35</td><td>18.52</td><td>3.06</td></tr><tr><td>HAR-CS</td><td>6.66</td><td>6.66</td><td>5.90</td><td>1.87</td></tr><tr><td>HAR-EMD</td><td>31.20</td><td>31.20</td><td>15.86</td><td>3.13</td></tr><tr><td>HAR-EMS</td><td>5.79</td><td>5.79</td><td>5.04</td><td>2.42</td></tr><tr><td>FD-C FD-CD</td><td>14.81</td><td>11.93</td><td>7.35</td><td>1.50</td></tr><tr><td>FD-CS</td><td>41.80 8.36</td><td>33.30 6.64</td><td>21.36 7.28</td><td>3.44 1.97</td></tr><tr><td>FD-EMD</td><td>38.94</td><td>27.65</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>18.05</td><td>3.54</td></tr><tr><td>FD-EMS</td><td>7.28</td><td>5.82</td><td>6.22</td><td>2.30</td></tr><tr><td>SR-C</td><td>14.81</td><td>14.81</td><td>7.35</td><td>1.66</td></tr><tr><td>SR-CD</td><td>41.80</td><td>41.80</td><td>21.36</td><td>2.68</td></tr><tr><td>SR-CS</td><td>8.36</td><td>8.36</td><td>7.28</td><td>1.99</td></tr><tr><td>SR-EMD</td><td>38.94</td><td>38.94</td><td>18.05</td><td>2.97</td></tr><tr><td>SR-EMS</td><td>7.28</td><td>7.28</td><td>6.22</td><td>2.43</td></tr></table>

In contrast, PULSE completes only 1,420–3,541 peerassisted updates across all scenarios, corresponding to an approximately 80–94% reduction with respect to OppAVG. This substantially lower message-exchange demand is a practical advantage for the resource-constrained wearable devices considered in this work, where repeated communication incurs energy and latency costs. Importantly, as shown by the performance results discussed next, this reduction does not come at the expense of personalization efectiveness.

![](images/012c3d835366af35673fca13354c58982f27b545e223ee6893f6f30adf2890c7.jpg)  
Fig. 3: Exploded view of the SensWear ring configuration: enclosure, flexible main board, curved battery, and interchangeable daughter boards.

## V. On-Device Evaluation

We evaluate PULSE’s feasibility on SensWear, an open-source modular wearable platform based on Zephyr RTOS [28], [29]. Its flexible main board (Fig. 3) integrates a Nordic nRF54L15 wireless SoC with a 128- MHz Arm Cortex-M33 processor, Bluetooth Low Energy (BLE) connectivity, 256 KB of RAM, and 1.5 MB of nonvolatile memory [30]. A BHI360 six-axis IMU provides inertial sensing, while 4 MiB of external non-volatile memory supports local storage of sensor samples and model parameters.

At the firmware level, PULSE learning operations are mapped to event-driven Zephyr work items. During an opportunistic exchange, the device requesting peer assistance acts as the initiator (node i), while the selected peer acts as the responder (node j). The initiator sends its personalized head $h _ { i }$ to the responder, which evaluates it on a private local minibatch and returns the resulting gradient $g _ { j }$ . The initiator then incorporates the received gradient into its local update, completing the peer-assisted interaction without exchanging raw sensor data.

## A. Experimental Protocol

All measurements use the same local model and hyperparameters adopted in the HAR simulations. Timing and energy are evaluated on two communicating SensWear devices, with five warm-up repetitions followed by 100 measured trials for each execution path, while instrumentation is kept outside the measured intervals to avoid afecting the results. Latency is characterized through the mean, median, interquartile range, and p5–p95 intervals to capture variability in wireless communication, while energy is derived from event-level current measurements relative to the corresponding pre-event baseline. Incremental flash and static RAM requirements are measured against an otherwise identical sensing-and-BLE firmware image without PULSE, isolating the memory overhead introduced by the framework.

## B. Power and Energy

Power consumption is measured using a Keysight 34465A digital multimeter connected in series with a current-limited 3.7-V supply at the SensWear battery input, while the peer device is powered independently. Current is sampled at 50 Hz, with the initiator and responder roles profiled in separate runs. For each event, baseline power is estimated over the 0.4 s preceding its detected onset, and incremental energy is obtained by integrating the power above this baseline over the following 4.1 s, including the return to idle. The resulting value represents source-side energy measured at the board input, while the responder energy is estimated assuming the same 3.7-V supply voltage.

As shown in Fig. 4a, the energy required by PULSE remains particularly limited across all profiled operations. Local adaptation is the most demanding operation, with a median incremental energy of 42.88 mJ (IQR 42.18– 43.43 mJ), whereas initiating an opportunistic exchange requires only 26.85 mJ (IQR 26.46–27.28 mJ). The responder incurs a considerably smaller median cost of 6.87 mJ (IQR 2.15–20.65 mJ), indicating that assisting a nearby device introduces only a modest additional energy demand. These results show that peer-assisted learning does not dominate the energy cost of on-device adaptation, despite involving both local computation and wireless communication. This behavior is also reflected in the current traces, where local adaptation sustains elevated consumption for a longer interval, while peer collaboration concentrates its energy demand in shorter computation and communication phases. The measured energy profile therefore supports the feasibility of selective peer assistance on battery-powered wearable devices, particularly when exchanges are performed only when considered beneficial by PULSE.

## C. Memory Usage

Flash and static RAM requirements are obtained from the linker maps of the PULSE firmware and an otherwise identical sensing-and-BLE baseline built from the same firmware revision. The diference between the two configurations is used to isolate the incremental memory footprint introduced by PULSE, with the linker-defined usable regions taken as the available device capacity.

As shown in Fig. 4b, the complete PULSE firmware occupies 561.8 KiB of flash (36.9% of the available 1,524 KiB) and 149.8 KiB of static RAM (58.5% of 256 KiB), leaving 962.2 KiB and 106.2 KiB available, respectively. Compared with the considered baseline, PULSE introduces an additional 211.0 KiB of flash and 91.9 KiB of RAM, showing that the complete learning framework fits within the memory resources of the wearable while retaining substantial capacity for other application functionality. The classification head contains 4,550 parameters and occupies 18.2 KB when serialized, with the same size required for an exchanged gradient. In contrast, the additional state required for peer selection is negligible, with the complete utility table for the configured eight peers requiring only 208 bytes. The remaining memory capacity confirms that PULSE can be practically deployed alongside the sensing and communication functionality of a resource-constrained wearable.

## D. Execution Time

Execution time is measured using the firmware’s 1-MHz wall counter, excluding subsequent logging operations. We profile both already-connected exchanges, which isolate the PULSE protocol, and discovery-to-update transactions that additionally include BLE link establishment. Each end-to-end path is evaluated over 100 trials, with latency distributions reported in Fig. 4c.

Local adaptation completes in 3539.6 ms at the median (3540.5 ms at p95), while an already-connected opportunistic exchange requires 3167.5 ms (3220.2 ms at p95). Including peer discovery and connection increases the median latency to 3428.4 ms, corresponding to an overhead of only 260.9 ms (8.2%) over the connected path. Most of this overhead is attributable to scanning and connection establishment, while peer selection and gradient processing introduce comparatively small computational costs. All 100 initiator trials successfully complete in both exchange configurations, showing that the complete PULSE interaction can be reliably executed directly on a wearable device within just a few seconds.

## VI. Conclusions and Future Work

This paper presented PULSE, an event-driven framework for Personalized Opportunistic Learning on resourceconstrained wearable devices. PULSE treats transient encounters as selective collaboration opportunities, combining utility-guided peer selection with gradient-level compatibility assessment. Across heterogeneous wearable applications and mobility conditions, PULSE consistently preserves or improves personalized performance while requiring substantially fewer peer-assisted exchanges than less selective opportunistic approaches. These results show that efective collaboration does not require exploiting every available encounter, highlighting the importance of selectivity in opportunistic settings. Moreover, the prototype implementation further demonstrates the feasibility of the proposed mechanisms on wearable-class hardware.

![](images/f4c368528296630a3f1b41fed06613b7c14846c5b0ef347e70a9026b7d0149be.jpg)  
(a)

![](images/9898c9d78256c4c2cda6b29a9ba26beb9c60f040c098cf27917a8722b3d81332.jpg)  
(b)

![](images/0f122002d07825713f310506f2d19afe22edb93255dbc2a76cd295314a0a7cb9.jpg)  
(c)  
Fig. 4: On-device feasibility of PULSE. (a) Energy distributions for local, initiator, and responder operations; (b) static memory footprint of the PULSE implementation; and (c) latency distributions measured on a SensWear device pair. Energy roles were measured separately and therefore do not represent paired exchanges.

Future work will extend PULSE towards more adaptive forms of opportunistic collaboration, where devices can dynamically select the knowledge to exchange and the most appropriate strategy for integrating it according to their current context, learning needs, and resource constraints. We also plan to investigate richer contextaware peer-utility models that explicitly account also for communication and energy costs, and to evaluate longterm personalized adaptation in larger-scale real-world deployments under naturally occurring mobility patterns.

## References

[1] A. Pantelopoulos and N. G. Bourbakis, “A survey on wearable sensor-based systems for health monitoring and prognosis,” IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 40, no. 1, pp. 1–12, 2010.

[2] F. Demrozi, G. Pravadelli, A. Bihorac, and P. Rashidi, “Human activity recognition using inertial, physiological and environmental sensors: A comprehensive survey,” IEEE Access, vol. 8, pp. 210 816–210 836, 2020. [Online]. Available: https://doi.org/10.1109/ACCESS.2020.3037715

[3] O. Durmaz Incel and S. Ö. Bursa, “On-device deep learning for mobile and wearable sensing applications: A review,” IEEE Sensors Journal, vol. 23, no. 6, pp. 5501–5512, 2023.

[4] J. Li, Y. Yang, Y. Chen, J. Zhang, Z. Lai, and L. Pan, “Dwlr: Domain adaptation under label shift for wearable sensor,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, K. Larson, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2024, pp. 4425–4433, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2024/489

[5] Y. Chang, A. Mathur, A. Isopoussu, J. Song, and F. Kawsar, “A systematic study of unsupervised domain adaptation for robust human-activity recognition,” Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., vol. 4, no. 1, pp. 39:1–39:30, 2020. [Online]. Available: https://doi.org/10.1145/3380985

[6] R. Presotto, G. Civitarese, and C. Bettini, “Federated clustering and semi-supervised learning: A new partnership for personalized human activity recognition,” Pervasive and Mobile Computing, vol. 88, p. 101726, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1574119222001390

[7] M. G. Campana and F. Delmastro, “Fedsub: Introducing classaware subnetworks fusion to enhance personalized federated learning,” IEEE Transactions on Artificial Intelligence, pp. 1– 15, 2026.

[8] P. Kairouz and H. B. McMahan, “Advances and open problems in federated learning,” Foundations and Trends in Machine Learning, vol. 14, no. 1-2, pp. 1–210, 06 2021.

[9] S. Lee, X. Zheng, J. Hua, H. Vikalo, and C. Julien, “Opportunistic federated learning: An exploration of egocentric collaboration for pervasive computing applications,” in 2021 IEEE International Conference on Pervasive Computing and Communications (PerCom), 2021, pp. 1–8.

[10] V. Romero, T. Matsui, Y. Matsuda, H. Suwa, and K. Yasumoto, “OPTIMIST: opportunistic federated transfer learning for mobile devices using sequential independent subnetwork training,” in IEEE International Conference on Pervasive Computing and Communications Workshops and other Afiliated Events, PerCom 2025 - Workshops, Washington DC, USA, March 17- 21, 2025. IEEE, 2025, pp. 183–188.

[11] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-eficient learning of deep networks from decentralized data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA, ser. Proceedings of Machine Learning Research, A. Singh and X. J. Zhu, Eds., vol. 54. PMLR, 2017, pp. 1273–1282.

[12] L. Yuan, Z. Wang, L. Sun, P. S. Yu, and C. G. Brinton, “Decentralized federated learning: A survey and perspective,” IEEE Internet of Things Journal, vol. 11, no. 21, pp. 34 617– 34 638, 2024.

[13] R. Mishra and H. P. Gupta, “Towards understanding the impact of participant and its wearable devices in federated learning,” IEEE Trans. Mob. Comput., vol. 24, no. 6, pp. 5003–5015, 2025. [Online]. Available: https://doi.org/10.1109/TMC.2025.3530818

[14] T. Tsouparopoulos and I. Koutsopoulos, “Implementation of federated learning on resource-constrained devices: Lessons learned,” in 2022 IFIP Networking Conference (IFIP Networking), 2022, pp. 1–6.

[15] A. Trotta, F. Montori, L. Ciabattini, G. Billi, L. Bononi, and M. D. Felice, “Edge human activity recognition using federated learning on constrained devices,” Pervasive Mob.

Comput., vol. 104, p. 101972, 2024. [Online]. Available: https://doi.org/10.1016/j.pmcj.2024.101972

[16] W. Ni, H. Ao, H. Tian, Y. C. Eldar, and D. Niyato, “Fedsl: Federated split learning for collaborative healthcare analytics on resource-constrained wearable iomt devices,” IEEE Internet Things J., vol. 11, no. 10, pp. 18 934–18 935, 2024. [Online]. Available: https://doi.org/10.1109/JIOT.2024.3370985

[17] X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, and J. Liu, “Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[18] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 5824–5836.

[19] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, “GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80. PMLR, 10–15 Jul 2018, pp. 794–803. [Online]. Available: https://proceedings.mlr.press/v80/chen18a.html

[20] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the multiarmed bandit problem,” Machine Learning, vol. 47, no. 2, pp. 235–256, May 2002.

[21] J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, “Human Activity Recognition Using Smartphones,” UCI Machine Learning Repository, 2013, DOI: https://doi.org/10.24432/C54S4K.

[22] H. Blunck, S. Bhattacharya, T. Prentow, M. Kjrgaard, and A. Dey, “Heterogeneity Activity Recognition,” UCI Machine Learning Repository, 2015, DOI: https://doi.org/10.24432/C5689X.

[23] A. Sucerquia, J. D. López, and J. F. Vargas-Bonilla, “Sisfall: A fall and movement dataset,” Sensors, vol. 17, no. 1, p. 198, 2017. [Online]. Available: https://doi.org/10.3390/s17010198

[24] P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven, “Introducing wesad, a multimodal dataset for wearable stress and afect detection,” in Proceedings of the 20th ACM International Conference on Multimodal Interaction, ser. ICMI ’18. New York, NY, USA: Association for Computing Machinery, 2018, p. 400–408. [Online]. Available: https://doi.org/10.1145/3242969.3242985

[25] J. Scott, R. Gass, J. Crowcroft, P. Hui, C. Diot, and A. Chaintreau, “Crawdad cambridge/haggle (v. 2009-05-29),” 2022. [Online]. Available: https://dx.doi.org/10.15783/C70011

[26] M. Musolesi and C. Mascolo, “A community based mobility model for ad hoc network research,” in Proceedings of the 2nd International Workshop on Multi-Hop Ad Hoc Networks: From Theory to Reality, REALMAN@MobiHoc 2006, Florence, Italy, May 26, 2006, M. Conti, J. Crowcroft, and A. Passarella, Eds. ACM, 2006, pp. 31–38. [Online]. Available: https://doi.org/10.1145/1132983.1132990

[27] A. E. F. Clementi, C. Macci, A. Monti, F. Pasquale, and R. Silvestri, “Flooding time in edge-markovian dynamic graphs,” in Proceedings of the Twenty-Seventh Annual ACM Symposium on Principles of Distributed Computing, PODC 2008, Toronto, Canada, August 18-21, 2008, R. A. Bazzi and B. Patt-Shamir, Eds. ACM, 2008, pp. 213–222. [Online]. Available: https://doi.org/10.1145/1400751.1400781

[28] SensWear, “SensWear: Open-source hardware and software repositories,” GitHub, 2026, accessed: Sep. 2, 2026. [Online]. Available: https://github.com/Sens-Wear

[29] Zephyr Project, “Zephyr Project,” https://www.zephyrproject.org/, accessed: Sep. 18, 2026.

[30] Nordic Semiconductor, “nRF54L15 system-on-chip,” Product page, 2026, accessed: Sep. 2, 2026. [Online]. Available: https://www.nordicsemi.com/Products/nRF54L15