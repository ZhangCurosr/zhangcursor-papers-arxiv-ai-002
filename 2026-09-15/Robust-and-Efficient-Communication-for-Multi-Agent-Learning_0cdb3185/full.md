# Robust and Efficient Communication for Multi-Agent Learning

Rafael Pina, Varuna De Silva and Corentin Artaud Institute for Digital Technologies   
Loughborough University London, United Kingdom r.m.pina, v.d.de-silva, c.artaud2@lboro.ac.uk

Abstract—Effective communication is a cornerstone of distributed intelligence in Multi-Agent Reinforcement Learning (MARL), yet ensuring that generated messages are both informative and robust to physical constraints remains a significant challenge. This paper introduces Multi-Agent Regularized Communication (MARC), a novel framework inspired by information-theoretic principles of conditional mutual information. MARC employs an attention-based architecture coupled with a unique message regularization mechanism designed to minimize uncertainty regarding future system states, thereby inducing the learning of highly representative communication protocols. Crucially, we evaluate MARC under stringent communication bottlenecks and lossy channels, simulating the realworld constraints of autonomous robotic networks and decentralized systems. Our results demonstrate that MARC significantly outperforms state-of-the-art methods in complex cooperative domains. Furthermore, we provide a deep analysis of message characteristics, proving that MARC maintains high operational performance even under significant data compression, offering a scalable path for deploying intelligent agents in resourceconstrained environments.

Index Terms—Multi-Agent Systems, Reinforcement Learning, Emergent Communication, Information Compression, Deep Learning

## I. INTRODUCTION

In distributed intelligent systems, agents often operate under conditions of partial observability, where critical environmental information is partitioned across the network and cannot be directly sensed by a single entity [1]–[3]. In such complex cooperative tasks, effective communication acts as a fundamental bridge, enabling agents to transcend local limitations by sharing encoded perceptions with their teammates to achieve a collective goal [4].

Recent advancements in Multi-Agent Reinforcement Learning (MARL) have positioned emergent communication as a pivotal research frontier [1], [2], [5], [6]. Most existing frameworks utilize the Centralized Training with Decentralized Execution (CTDE) paradigm [7], [8]. While CTDE provides a powerful training foundation, relying on a centralized oracle is often unfeasible in practical field deployments due to latency, privacy, or infrastructure constraints [9], [10]. Communicationenabled decentralized execution offers a more robust alternative; by broadcasting learned message encodings rather than raw observation data, agents can maintain high-level coordination without the need for a global controller [11], [12]. When communication is available, under the CTDE paradigm both execution and training can be improved since agents can broadcast what they see or know at a certain moment to the rest of the teammates (as portrayed in Fig. 1). However, a significant challenge remains: ensuring these learned messages are sufficiently representative to drive meaningful cooperation while remaining stable throughout the non-stationary learning process.

![](images/fe17d4900ff660a77745fcf73deb9759e5d5d530e9e3476d87007489307c9d88.jpg)

![](images/644d4d2ff5489bb7e85a01d7de2c232e1a98d8d730f496db54789b97a5532549.jpg)  
(a) Lumberjacks  
(b) TrafficJunction  
Fig. 1. Representation of two partially obserable environments where the agents can only see their surroundings and thus will strongly benefit if they receive information from the others about their local perceptions of parts of the environment that are far away (from [13]).

Furthermore, the physical reality of communication channels—specifically limited bandwidth and signal noise— presents a major hurdle for autonomous systems [10], [14], [15]. Just as wireless communication protocols must compress data to fit within restricted spectral bands [16], intelligent agents must learn to generate information-dense messages that survive transmission through constrained or lossy channels.

In this paper, we propose Multi-Agent Regularized Communication (MARC), a framework designed to produce highly representative communication protocols within the CTDE paradigm. MARC is modular and compatible with diverse value-function factorization methods. Our approach leverages an attention-based architecture enhanced by a novel information-inspired regularizer. This mechanism stabilizes the learning process by coaxing agents to learn messages that minimize the uncertainty regarding their future underlying observations, effectively maximizing the utility of the shared information.

Addressing the necessity of bandwidth efficiency, we also explore a novel perspective on message compression via the Discrete Cosine Transform (DCT). While frequently utilized in classical signal processing, the DCT has been largely overlooked in recent MARL literature. We demonstrate that information-based lossy compression can significantly reduce message size without compromising system-level performance, offering a scalable solution for resource-constrained hardware. Other related communication methods often fail to deal with compression and learn efficient communication strategies [15].

In summary, the primary contributions of this work are as follows:

• A Novel Regularized Architecture: We introduce MARC, featuring a unique message regularizer that enhances agent performance across diverse environments and ensures the generation of highly representative communication protocols.

• Robustness under Compression: We analyze the effects of lossy message compression using the DCT, demonstrating its efficacy in reducing communication overhead in complex MARL tasks.

• Behavioral & Message Analysis: We provide an indepth empirical study of learned message structures under both ideal and lossy conditions, showcasing the adaptive behaviors of the agents.

• Theoretical Foundations: We provide the informationtheoretic grounding and motivations that underpin the proposed regularization mechanism.

The rest of this paper is organised as follows: in section II we introduce other relevant works that relate to this paper, and in section III we present the key principles to provide the needed context to the concepts introduced in this work. In section IV we present in detail the proposed methods in this paper, followed by the experiments in section V. Finally, in section VI we present the conclusions of this paper, together with some potential avenues for further research.

## II. RELATED WORK

MARL has been well studied in the recent past [1], [17]– [20]. Particularly in cooperative MARL, value function factorisation methods represent a branch of popular algorithms whose core objective is to learn a decomposition of a joint Qfunction into agent-wise Q-functions. Numerous approaches that follow this configuration have shown outstanding results in multiple complex scenarios [20]–[23]. Importantly, these methods operate under the well-known CTDE paradigm. While this configuration represents the middle of the learning spectrum, the other two ends have also been studied, but it has been discussed that neither full centralisation nor full decentralisation can achieve as good results [18].

As an alternative to the conventional methods, communication in MARL has gained the attention of the scientific community [1], [2], [5], [24]–[26]. At its core, the key difference in these methods when compared to conventional approaches is that agents can share something about their experience of the environment, both during training and execution. In [2] the authors have introduced one of the very first communicationbased MARL methods. In simple terms, it is shown how two agents can solve tasks where they must communicate some of what one knows but the other doesn’t. In [6], the authors show the effect of communication in more complex scenarios with more agents. In particular, it is shown that, when agents communicate, their level of efficient cooperation increases in scenarios with more complex objectives, such as traffic junction negotiations. As these approaches became more popular, more complex methods started to arise. These consider factors such as what, whom or when to communicate. For example, in [24] it is proposed an attention-based mechanism that allows the agent to decide to whom they should send their messages. In the case of [5], the authors propose an improved messaging mechanism that consists of signing the messages, allowing to target specific agents with customised information. Also in [27], the authors propose a communication method that uses message fingerprints and a different way of aggregating all the messages of the agents. In [28] it is also shown a different way of aggregating messages in a manner that represents them with information that is more relevant for the policies. That is achieved through an aggregation of all the observations with the aim of predicting a global state representation. What is shared can also be important, as demonstrated in works such as [1] or [4], where agents that share their intentions achieve better performances.

Still involving MARL but on a slightly different scope, language discovery in communication games has also been aim of studies. Usually in games that involve a speaker and a listener agent, the kind of language that emerges during the communication process has been studied in works like [29], where the agents learn a task-specific vocabulary during training that they use to solve the tested games. Also in [30] or [31] the authors evaluate the interpretability of the languages learned by the agents in terms of how humans can interpret them. Another step towards interpretability is to ground language, as it is studied in [32] where the authors use an autoencoder module that grounds language by reconstructing observations. While this deeper analysis becomes more challenging as more agents are used, it is still important to analyse what the agents devise to communicate in order to understand their learning process and how they understand the imposed tasks.

Compressing messages in MARL has also been aim of studies. For instance, [33] proposes a communication architecture that can schedule messages in a way such that it can optimally operate under bandwidth contrainsts. This type of bandwidth limitations are relevant in MARL studies [14], [15], which might benefit from message compression in communication. In this work we tackle the problem from a different perspective and study how the DCT can help in compression for MARL.

## III. BACKGROUND

A. Decentralised Partially Observable Markov Decision Processes (Dec-POMDPs)

In this work, we model the learning problems following Decentralised Partially Observable Markov Decision Processes (Dec-POMDPs) [34]. The Dec-POMDP can be defined as a tuple $G = \langle S , A , O , Z , P , r , \gamma , N \rangle$ , where $s \in S$ represents the current state of the environment. From the current state, local observations $o _ { i }$ can be derived according to a function $O ( s , i ) : S \times N \to Z$ for a certain agent $i \in \mathcal { N } \equiv \{ 1 , \ldots , N \}$

Additionally, the agents also maintain an action-observation history $\tau _ { i } \in \mathcal { T } \equiv ( Z \times A ) ^ { * }  \{ \tau _ { 1 } , \dots , \tau _ { N } \}$ . When in a given state, each agent performs an action $a _ { i } \in A$ where A is the action space, forming a joint action $a = \{ a _ { 1 } , \ldots , a _ { N } \}$ that is executed in the current state s of the environment, and from which it results a reward that is shared by the entire team, $r ( s , a ) : S \times A \to \mathbb { R }$ . After the actions are executed, the environment transits to a next state $s ^ { \prime }$ according to a probability function that models the dynamics of the environment, $P ( s ^ { \prime } | { a } , { s } ) : S \times A \times S  [ 0 , 1 ]$ . Usually, the actions of the agents are controlled by a policy $\pi _ { i } ( a _ { i } | \tau _ { i } ) : \mathcal { T } \times A \to [ 0 , 1 ] ,$ but in the considered context of communication in MARL, the policy is instead conditioned not only on $\tau _ { i } ,$ , but also on a set of incoming messages from the other agents $m _ { - i } ,$ meaning that the corresponding policy that controls the actions of the agents can be written as $\pi _ { i } ( a _ { i } | \tau _ { i } , m _ { - i } )$ (where −i refers to all except i). During learning, the joint objective of the agents is to maximise an action-value function $Q _ { \pi } ( s _ { t } , a _ { t } ) = \mathbb { E } _ { \pi } [ R _ { t } | s _ { t } , a _ { t } ]$ where $\begin{array} { r } { R ^ { t } \ = \ \sum _ { k = 0 } ^ { \infty } \gamma _ { k } r _ { t + k } } \end{array}$ is the discounted return with a discount factor $\gamma \in [ 0 , 1 )$

## B. Centralised Training with Decentralised Execution (CTDE) and Communication in MARL

Within MARL, centralised training with decentralised execution (CTDE) is a popular paradigm that allows the agents to be trained in a centralised manner, but they must execute their policies in a decentralised way [18], [19], [35]. Communication-based methods can be easily integrated into this framework. The key difference from simple CTDE is that, with communication, the agents can broadcast information to others not only during training but also during execution. This means that, despite the learned policies being decentralised, the agents can receive encoded messages that represent what their teammates see or sense at a certain timestep. While in the fully centralised setting there is a common oracle that sees everything in the environment, a communication setting can be seen as a more realistic choice since communication is done on a peer-to-peer basis where each agent is responsible for broadcasting its own messages. Thus, they are not dependent on a central unit that has the big - and often unrealistic - advantage of observing everything at the same time.

Intuitively, communication in MARL should improve the performances of the agents when it is combined with other base methods, as shown in works such as [1]. If the messages learned are informative enough, they should be useful for the agents to learn to “talk” with each other and improve their cooperative strategies. However, if the messages learned are not adequate, this might result in an overload for the learning networks and harm the performances of the agents.

## C. Value Function Factorisation in MARL

In cooperative MARL, value function factorisation methods form a group of powerful algorithms to solve complex MARL tasks. The key idea of these methods is to learn a way of decomposing a joint action-value function into agent-wise functions [18],

$$
\begin{array} { r } { Q _ { t o t } \left( \tau , a \right) = f \left( Q _ { i } \left( \tau _ { i } , a _ { i } ; \theta _ { i } \right) \right) , \forall i \in \{ 1 , \ldots , N \} , } \end{array}\tag{1}
$$

where f here represents a certain function that mixes the individual functions into a joint function $Q _ { t o t }$ . An efficient decomposition of the joint action-value function should be done in a way that satisfies the Individual-Global-Max (IGM) condition [36]. This condition states that the set of local optimal actions should also maximise the joint Q-function. This can be formalised as

$$
\operatorname * { a r g m a x } _ { a } Q _ { t o t } \left( \tau , a \right) = \left( \begin{array} { c } { \operatorname * { a r g m a x } _ { a _ { 1 } } Q _ { 1 } ( \tau _ { 1 } , a _ { 1 } ) } \\ { \vdots } \\ { \operatorname * { a r g m a x } _ { a _ { N } } Q _ { N } ( \tau _ { N } , a _ { N } ) } \end{array} \right) .\tag{2}
$$

In [18], the authors introduce Value Decomposition Networks (VDN) as a way of factorising the joint $Q _ { t o t }$ as the sum of the individual Q-functions. Later on, QMIX [21] proposes a new non-linear way of factorising the $Q _ { t o t }$ that extends the range of functions that can be represented by VDN to a family of monotonic functions. Both these factorisation methods are sufficient to satisfy (2).

In these methods, the loss used to update the networks is based on the temporal difference loss as described in DQN [37] that uses a replay buffer and a target network, but here with respect to a $Q _ { t o t }$ . This loss can be formalised as

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { b \sim B } \left[ \left( r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { t o t } ( \tau ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) - Q _ { t o t } ( \tau , a ; \theta ) \right) ^ { 2 } \right] ,\tag{3}
$$

for some sample b that is sampled from the replay buffer B, and where $\theta$ and $\theta ^ { - }$ are the parameters of the learning network and of a target network, respectively. In this paper, we use VDN and QMIX to demonstrate our communication approach on top of standard MARL approaches that don’t initially use communication.

## D. Discrete Cosine Transform (DCT)

In the fields of data compression and signal processing, the Discrete Cosine Transform (DCT) [38] is a certain function based on a sum of cosine functions that processes a sequence of data points and encodes them into a compressed representation. By doing so, it is possible to achieve a much smaller representation of the data in terms of size occupied. For simplicity, we introduce here only the expression for the type II of the DCT (that is the most common form and the one used in this paper) that, generally, can be done following [39]

$$
X ( k ) = 2 \sum _ { n = 0 } ^ { C - 1 } x ( n ) \mathrm { { c o s } } \left( { \frac { \pi ( 2 n + 1 ) k } { 2 C } } \right) , 0 \leq k \leq C - 1 ,\tag{4}
$$

where $X ( k )$ represents the $k ^ { t h }$ transform of the DCT, and $x ( n )$ denotes the sequence of values to be compressed, with size C. In the context of this paper, the DCT is applied to the last dimension of the vectors containing the messages of the agents, forming a compressed representation of their messages. Using this smaller representation is important when we consider channel sizes or other constraints, through which only compressed representations can be sent. When this representation arrives at the other end of the communication channel, the inverse of the function is applied, and the message can be recovered with some information loss. We hypothesise that, if this information loss is not too heavy, the agents can still learn and benefit from communication in MARL. This decompressing process can be done by inverting Eq. (4). The inverse can be calculated following [39]

$$
x ( n ) = \frac { 1 } { C } \left[ \frac { X ( 0 ) } { 2 } + \sum _ { k = 1 \atop 0 \leq n \leq C - 1 . } ^ { C - 1 } X ( k ) c o s \left( \frac { \pi ( 2 n + 1 ) k } { 2 C } \right) \right] ,\tag{5}
$$

The DCT can be used both as a lossy and as a lossless compression method, depending on the parts of the messages that are encoded. In this paper, we consider the case of lossy compression, since for lossless compression to be achieved the messages are encoded but without reducing their sizes, which becomes redundant when we consider limited communication channels.

## IV. METHODS

A. Multi-Agent Regularized Communication (MARC) in MARL

In this section, we propose MARC, a new architecture for efficient communication in MARL. The idea is to build an inter-agent communication architecture that can learn meaningful messages to ensure cooperation in complex MARL tasks. In the proposed architecture, MARC starts by using an attention module to learn messages that are generated from the local observations of the agents. This mechanism starts by encoding the observations that are then given as an initial message $m ^ { \prime }$ to an attention module, in order to learn their relative importance. These values are embedded into $k \in \mathbb { R } ^ { d _ { k } } , v \in \mathbb { R } ^ { d _ { k } } , q \in \mathbb { R } ^ { d _ { k } }$ , where $d _ { k }$ is the embedding dimension (as shown in Fig.2). As such, we define the keys, queries, and values, at each timestep t, for the attention operations as

$$
{ k _ { t } } = \left[ { { W _ { K , 1 } } \tau _ { 1 } ^ { t } , \dots , { W _ { K , i } } \tau _ { i } ^ { t } , \dots , { W _ { K , N } } \tau _ { N } ^ { t } } \right] ,\tag{6}
$$

$$
\boldsymbol { v } _ { t } = \left[ W _ { V , 1 } \tau _ { 1 } ^ { t } , \dots , W _ { V , i } \tau _ { i } ^ { t } , \dots , W _ { V , N } \tau _ { N } ^ { t } \right] ,\tag{7}
$$

$$
q _ { t } = \left[ W _ { Q , 1 } m _ { 1 } ^ { \prime t } , \dots , W _ { Q , i } m _ { i } ^ { \prime t } , \dots , W _ { Q , N } m _ { N } ^ { \prime t } \right] ,\tag{8}
$$

where $W _ { Q , i } , W _ { K , i } , W _ { V , i }$ are trainable weight matrices, and where $m ^ { \prime t }$ corresponds to the initial message embeddings that will be refined (as in Fig. 2). As a result, the attention weights calculated for an agent i from the messages and observations in our approach can be formalised as

$$
\alpha _ { i j } = \frac { \exp ( \phi \cdot ( q _ { i } ^ { t } \cdot \boldsymbol { k } _ { j } ^ { t ^ { T } } ) ) } { \sum _ { \boldsymbol { x } \in \mathcal { N } } \exp ( \phi \cdot ( q _ { i } ^ { t } \cdot \boldsymbol { k } _ { x } ^ { t ^ { T } } ) ) } ,\tag{9}
$$

where $\phi$ is a scaling factor. These weights are further used to calculate a new aggregated attention-encoded message $m _ { i } ^ { t } .$

$$
m _ { i } ^ { t } = \sum _ { j = 1 } ^ { N } \alpha _ { i j } v _ { j } ^ { t } ,\tag{10}
$$

where the weights $\alpha _ { i j }$ define the relationship between agents $i$ and $j ,$ and hence the importance of value $\ v { v } _ { j } ^ { t }$ . The intuition behind the choice of keys, queries and values, is that, by relating the messages to multiple different latent representations of the observations, the messages learned will be able to capture more relevant information from the observations.

Despite the recent success of attention-based architectures for multi-agent communication in complex scenarios, we begin with the hypothesis that the messages originated by our base architecture are not rich enough to learn complex environments. In this sense, we propose a message regularizer in our architecture. The key intuition is that, if the messages generated by the agents help to predict their own next observations, it means that these messages are likely to contain more meaningful information about their individual observations. We hypothesise that the uncertainty associated with the future values of the observations is reduced when we use both the previous values of the messages and the observations, when compared to using only the previous values of the observations. Motivated by the concepts of conditional entropy, our hypothesis is formalised in the following Theorem 1, when we consider a certain set of messages that contains information about the observations.

Theorem 1. Let m be a certain encoding of the observations o and $m ^ { - }$ and $o ^ { - }$ represent the previous values of each, respectively. We have that $H ( o | o ^ { - } ) \geq H ( o | o ^ { - } , m ^ { - } )$

Proof. Since m is a certain encoding of $^ { O , }$ it is true that it contains information about $o .$ The mutual information between the observations and the previous messages when conditioned on the previous observations can be written as

$$
I ( o , m ^ { - } | o ^ { - } ) = H ( o , o ^ { - } ) + H ( m ^ { - } , o ^ { - } ) - H ( o , o ^ { - } , m ^ { - } ) - H ( o ^ { - } )\tag{11}
$$

$$
= H ( o | o ^ { - } ) - H ( o | o ^ { - } , m ^ { - } )\tag{12}
$$

Since the mutual information is always non-negative, from the above we can write:

$$
H ( o | o ^ { - } ) - H ( o | o ^ { - } , m ^ { - } ) \ge 0\tag{13}
$$

$$
H ( o | o ^ { - } ) \geq H ( o | o ^ { - } , m ^ { - } ) ,\tag{14}
$$

meaning that conditioning o both on $o ^ { - }$ and $m ^ { - }$ reduces the uncertainty linked to the observations when compared to conditioning only on $o ^ { - }$ □

Motivated by Theorem 1, we will define a message regularizer that aims to predict the next observations of the agents, given the previous messages received together with the previous observations. As such, we use a recurrent encoder that receives the previous messages alongside the previous observations, $[ o _ { i } ^ { - p } , m _ { i } ^ { - p } ]$ with values up to $T - p ,$ , and predicts the next $p$ observations ahead $o _ { i } ^ { + p }$ , where $p$ denotes the number of timesteps to predict ahead of each timestep t, and $T$ the total length of the episode. For our experiments, we predict only one timestep ahead of each t. Let $g ( \cdot ; \theta _ { r } )$ here denote a certain neural network with parameters $\theta _ { r }$ composed of an LSTM module that estimates the next values of the observations of the agents given the previous messages and the previous observations (as described in Fig. 2). The observations predicted by this network for the timesteps ahead can be given by $o _ { i } ^ { + p ^ { \prime } } \ : \stackrel {  } { = } \ : g ( [ o _ { i } ^ { - p } , m _ { i } ^ { - p } ] ; \theta _ { r } )$ . The predicted outputs of this network can then be used to calculate a second loss that will auxiliate our learning problem, as described in

![](images/90786ed113d7a7fda301bdb77807583b3e0a8d75bd2ed039241e3e54a4eaf876.jpg)  
Fig. 2. Architecture of the proposed communication method for MARL. The proposed method can be used together with any value function factorisation method (whose mixer is represented in the figure in the block mixer) and uses parameter sharing. The pink diamonds represent the inverse DCT that is applied to decompress the messages that are compressed with the DCT (yellow diamond) after they are computed by the communication network (yellow). Note that both the DCT and IDCT blocks are only used when we analyse the effect of compression in section V-D. In the main experiments, these blocks are not applied.

$$
\mathcal { L } _ { m } = \frac { 1 } { T - p } \sum _ { t = 1 } ^ { T - p } \lVert o ^ { + p } - o _ { i } ^ { + p ^ { \prime } } \rVert _ { 2 } ^ { 2 } ,\tag{15}
$$

for an agent i, where $T - p$ denotes the number of predicted values in the vector, given that an episode lasts T timesteps and $g ( \cdot ; \theta _ { r } )$ is predicting k timesteps ahead of each timestep t. For the proposed method, this additional loss is used alongside the loss described in Eq. (3), resulting in the overall objective of minimising, with respect to τ, the following loss

$$
\begin{array} { l } { \displaystyle \mathcal { L } \big ( \theta , \theta _ { c } , \theta _ { r } \big ) = \sum _ { b = 1 } ^ { B } \Bigg [ \big ( y _ { t o t } - Q _ { t o t } \big ( \tau , a ; \theta \big ) \big ) ^ { 2 } } \\ { \displaystyle \qquad + \frac { 1 } { T - p } \sum _ { t = 1 } ^ { T - p } \| \tau ^ { + p } - g ( \tau ^ { + p } , m ^ { + p } ; \theta _ { r } ) \| _ { 2 } ^ { 2 } \Bigg ] , } \end{array}\tag{16}
$$

for a batch of samples $B ,$ and where $\begin{array} { r l r } { y _ { t o t } } & { { } = } & { r \ + } \end{array}$ $\gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { t o t } ( \tau ^ { \prime } , a ^ { \prime } ; \theta ^ { - } )$ , for the parameters of a network and the respective target, θ and $\theta ^ { - }$ , the parameters of the communication network $\theta _ { c } ,$ and the parameters of the regularizer module $\theta _ { r }$ . Ahead we demonstrate how this message regularizer positively affects the learning process of the agents. Fig. 2 depicts the architecture of the proposed method as described in this section (note that the DCT and IDCT blocks are represented for completeness and are not used when compression is not being analysed). Importantly, we build our communication method on top of value function factorisation methods and adopt the famous parameter sharing convention [40]. As such, our method can be easily integrated with any existing value function factorisation method, to which we add our regularizer objective without affecting the convergence of the algorithm, as shown in Theorem 2 below. For simplicity of notation, we do not consider the history of the agents in this demonstration. While this is not the case in the experiments, for the theorem we assume tabular conditions.

Theorem 2. Assuming that $\begin{array} { r } { 0 \le \alpha _ { k } < 1 , \sum _ { k } \alpha _ { k } = \infty , } \end{array}$ $\textstyle \sum _ { k } \alpha _ { k } ^ { 2 } < \infty ,$ , updating the learning problem following the rule

$$
\begin{array} { l } { { \displaystyle Q _ { t o t } ^ { k + 1 } ( o _ { t } , a _ { t } ) = Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } \Biggl [ r _ { t } + \gamma m a x _ { a } Q _ { t o t } ^ { k } ( o _ { t + 1 } , a ) } } \\ { { \displaystyle ~ - Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \frac { 1 } { T - p } \sum _ { t = 1 } ^ { T - p } \lVert o ^ { + p } - g _ { k } ( o ^ { + p } , m ^ { + p } ; \theta _ { r } ) \rVert _ { 2 } ^ { 2 } \Biggr ] . } } \end{array}\tag{17}
$$

will make $Q _ { t o t }$ converge to a certain value, such that $| | Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) - Q _ { t o t } ^ { * } ( o _ { t } , \bar { a } _ { t } ) | | \leq \lambda ( T - p ) G ,$ , as $k  \infty$ and $\| o ^ { + p } - g _ { k } \big ( o ^ { + p } , m ^ { + p } ; \theta _ { r } \big ) \| _ { 2 } ^ { 2 } \leq G .$

Proof. Let $y _ { t o t } = r _ { t } + \gamma m a x _ { a } Q _ { t o t } ^ { k } ( o _ { t + 1 } , a )$ for conciseness. Let $U _ { k } ( \cdot )$ denote the term $\begin{array} { r } { \sum _ { t = 1 } ^ { T - p } \lVert o ^ { + p } - g _ { k } ( o ^ { + p } , m ^ { + p } ; \theta _ { r } ) \rVert _ { 2 } ^ { 2 } } \end{array}$ By rearranging the previous equation, we can write

$$
\begin{array} { l } { { Q _ { t o t } ^ { k + 1 } ( o _ { t } , a _ { t } ) } } \\ { { \ } } \\ { { \displaystyle = Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } \left[ y _ { t o t } - Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \frac { 1 } { T - p } U _ { k } ( \cdot ) \right] } } \\ { { \ } } \\ { { \displaystyle = Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } \left[ y _ { t o t } - Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) - ( p - T ) ^ { - 1 } U _ { k } ( \cdot ) \right] } } \\ { { \ } } \\ { { \displaystyle = Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } \left[ y _ { t o t } - Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) - \lambda U _ { k } ( \cdot ) \right] \qquad ( 1 8 ) } } \end{array}
$$

for $\lambda = ( p - T ) ^ { - 1 }$ . Similarly to [25] and based on [41], we can now write the equation as a function of stochastic dynamic processes,

$$
\delta ^ { k + 1 } ( o _ { t } , a _ { t } ) = ( 1 - \alpha _ { k } ) \delta ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } \left[ F _ { k } ( o _ { t } , a _ { t } ) - \lambda U _ { k } ( \cdot ) \right] ,\tag{19}
$$

where $\begin{array} { r l r } { \delta ^ { k } ( o _ { t } , a _ { t } ) } & { { } = } & { Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) - Q _ { t o t } ^ { * } ( o _ { t } , a _ { t } ) } \end{array}$ and $F _ { k } ( o _ { t } , a _ { t } ) = r _ { t } + \gamma m a x _ { a } Q _ { t o t } ^ { k } ( o _ { t + 1 } , a ) - Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } )$ . This

can be decomposed into random processes according to $\delta ^ { k + 1 } ( o _ { t } , a _ { t } ) = \dot { \delta } _ { 1 } ^ { k + 1 } \big ( o _ { t } , a _ { t } \big ) + \delta _ { 2 } ^ { k + 1 } \big ( \dot { o _ { t } } , a _ { t } \big )$ , resulting in

$$
\delta _ { 1 } ^ { k + 1 } ( o _ { t } , a _ { t } ) = ( 1 - \alpha _ { k } ) \delta _ { 1 } ^ { k } ( o _ { t } , a _ { t } ) + \alpha _ { k } F _ { k } \big ( o _ { t } , a _ { t } \big )\tag{20}
$$

$$
\delta _ { 2 } ^ { k + 1 } ( o _ { t } , a _ { t } ) = ( 1 - \alpha _ { k } ) \delta _ { 2 } ^ { k } ( o _ { t } , a _ { t } ) - \alpha _ { k } U _ { k } ( o _ { t } , a _ { t } ) .\tag{21}
$$

From [42] it is known that the first process converges to zero w.p. 1. Similarly to [25], the second process can be written as

$$
\delta _ { 2 } ^ { k + 1 } ( o _ { t } , a _ { t } ) \leq ( 1 - \alpha _ { k } ) | | \delta _ { 2 } ^ { k } ( o _ { t } , a _ { t } ) | | + \alpha _ { k } \lambda ( T - p ) G ,\tag{22}
$$

which means that (17) converges to a number greater than zero. Thus, $Q _ { t o t }$ will converge to an optimal value as the iterations of the update tend to infinity, i.e.,

$$
\begin{array} { r l } & { | | Q _ { t o t } ^ { k } ( o _ { t } , a _ { t } ) - Q _ { t o t } ^ { * } ( o _ { t } , a _ { t } ) | | = | | \delta ^ { k } ( o _ { t } , a _ { t } ) | | } \\ & { \phantom { | | } = | | \delta _ { 1 } ^ { k } ( o _ { t } , a _ { t } ) + \delta _ { 2 } ^ { k } ( o _ { t } , a _ { t } ) | | } \\ & { \phantom { | | } \le | | \delta _ { 1 } ^ { k } ( o _ { t } , a _ { t } ) | | + | | \delta _ { 2 } ^ { k } ( o _ { t } , a _ { t } ) | | } \\ & { \phantom { | | } \le \lambda ( T - p ) G . \phantom { | | } ( 2 3 ) } \end{array}
$$

Hence, this means that the proposed method will still lead to convergence of the learning process, as the number of iterations $k$ tends to infinity. □

## B. Message Compression for Lossy Communication in MARL

In this work, we also intend to study whether communication can still be done efficiently when the messages exchanged are, for example, lost, or dropped during the process. Hence, instead of simply zeroing or naively cutting values of the messages, we use the DCT, a popular lossy compression method. We note that this method for message compression does not require changing the sizes of any of the agent networks, since the compression only applies to a potential communication channel, and the messages are reconstructed to the original size when they reach the other communication end.

In our architecture, each agent will generate its messages according to the method described in the previous subsection, and then, in the results presented in subsection V-D, the generated messages will be compressed using the DCT, as described in Eq. (4). In this work, we focus on lossy compression, i.e., each agent compresses the message in a way that will make it smaller in size, reducing the potential communication overhead. When the messages arrive at the destination, these are decompressed using the inverse of the DCT (IDCT, as described in Eq. (5). This is depicted in the architecture in Fig. 2). When we consider message compression, the change in the architecture comes from the compression of the message after it is generated (DCT in the yellow diamond of the figure), and then it is decompressed in the destination (IDCT, illustrated by the pink diamonds in the figure). Finally, in section V-D we use the result of this process as input of the agents, together with their own observations, in the same way that was described before, but now the messages are compressed in the source and then decompressed in the endpoint. This results in the input $[ \tau _ { i } , m _ { - i } ^ { * } ]$ for agent i, where $m _ { - i } ^ { * }$ here represents the decompressed messages coming from the other agents.

## V. EXPERIMENTS AND RESULTS<sup>1</sup>

In this section, we present the experiments carried out to evaluate the performance of the proposed MARC. While the proposed communication method can be built on top of any value function factorisation method in MARL, in this paper we apply it together with the architectures of QMIX [21] and VDN [18], two popular value function factorisation methods in MARL, as introduced in subsection III-C. We demonstrate the performances of MARC in comparison with the vanilla methods without communication, and three popular communication methods, MASIA [28], COMMNET [6], and TARMAC [5]. Note that, in the results in V-B, there is no compression, i.e., the DCT and IDCT blocks in Fig. 2 are not used. These are only used when investigating message compression in V-D. We start by presenting the experimental details and hyperparameters below.

## A. Hyperparameters and Implementation Details

Our experiments were executed in a desktop computer with an 18-core Intel Core i9-10980XE CPU, 256GB of RAM, and 3 NVIDIA A6000 GPUs. In the experiments carried out in this paper, the agents are controlled by a deep recurrent Qnetwork that uses a GRU (gated recurrent unit) with width 64. Additionally, in QMIX the hidden layers used by the mixer have size 32. All the experimented methods share the learning networks, following the parameter sharing convention widely adopted in the literature. This allows to speed up learning but requires that an agent ID is added to the inputs of the agents so that the network can differentiate them. Regarding the communication networks in MARC, we set the size of all the embeddings of the attention mechanism to 64. The length of the messages that each agent produces at each timestep is defined as 10. When using compression, a value of, for example, 40% of compression will naturally result in messages of size 6, and so on. Regarding TARMAC and MASIA, we use them with QMIX as the mixer.

We use a replay buffer to store experiences with size 5000, from which minibatches with size of 32 episodes are sampled for training. The replay buffer is updated over time. To take actions, the exploration-exploitation trade-off of the agents follows the epsilon-greedy method, with the value epsilon, ϵ, starting at 1. This value anneals gradually throughout 50000 training episodes down to a minimum of 0.05. The value of the discount factor γ is set to 0.99. The parameters of the target networks are updated every 200 episodes. All the networks are trained using the RMSProp optimization algorithm, with a learning rate $\alpha = 5 \times 1 0 ^ { - 4 }$

## B. Main Results: Performance in SMAC and PredatorPrey

Fig. 3(a) illustrates the performances in the 3s vs 5z SMAC scenario [43]. Considering that this is the simplest scenario out of all the experimented ones, it was expected that all the methods would be able to solve it somewhat easily. Particularly in the case of QMIX, we can see that the agents benefit from communication, making them solve the task sooner. However, in the case of VDN, it doesn’t show to be as useful. This suggests that, for this scenario, communication might not be as important as in other cases to learn efficient strategies, although convergence is still achieved quickly.

![](images/37c52ab12d76c04dd1f49cc6048b81c8424121e87706e6c90adb02728e33c23d.jpg)  
(a) 3s vs 5z

![](images/002ac591fc83d8cfec18eb2c809463c51281eae7f20dc3f0d6ca975a93cbc708.jpg)  
(b) 2c vs 64zg

![](images/13314b7ecd863cce63e60774b129efd9cf08c7f8a97b6fa7c0ee7111e81b5ce8.jpg)  
(c) MMM2

![](images/ae4ee6fe930a0e83a7237d59d636ab8a9d1e00f2064689bfec2f28c62f6427b5.jpg)  
(d) 1o2r vs 4r

Fig. 3. Performance of the attempted methods in the Starcraft environments. The plots depict the average win rates of the agents over training time. (a) In 3s vs 5z 3 stalker units must defeat a team with 5 zealot units. (b) In 2c vs 64zg, the agents are 2 colossi playing against 64 zerglings. (c) MMM2 where 1 medivac (healing unit), 2 marauders and 7 marines play against 1 medivac, 3 marauders and 8 marines. (d) In 1o2r vs 4r, 1 overseer and 2 roaches must defeat 4 reapers, where all units spawn at random points and only the overseer knows the enemy location (as in [28]).  
![](images/4b8a60ebdc6913f07493b6f7138fbddeb633ada1451d61ab71183e0754298ef5.jpg)  
(a) p = −0.50

![](images/98facd40cbf609f3d84aadeff72f6591417e372f9b4b9dfaa97fdc48b853fd54.jpg)  
(b) p = −0.75

![](images/08ab545248e7847c873a07698451ece429515fd50a7d90033d97541eacfbd78a.jpg)  
(c) p = −1.0  
Fig. 4. Performance of the attempted methods in the PredatorPrey environment for different levels of cooperation penalty. The plots depict the average rewards of the agents over training time.

Yet, in the other more complex scenarios communication proves to have a stronger impact. In Fig. 3(b), we can see that, for 2c vs 64zg, MARC enables the agents to learn the task much faster and at a higher level of performance. Both QMIX and VDN show much-improved performances when combined with MARC, with particular emphasis on VDN+MARC, where the improvement is outstanding. In 1o2r vs 4r (as used in works like [28], [44]) (Fig. 3(d)) we observe a similar scene, where QMIX+MARC stays above the others, although all of them achieve average winning rates. In the case of MMM2, in Fig. 3(c) we can also see benefits of communication, although these are not as prominent as in the case of the previous scenarios. The effect of communication becomes evident mostly when we look at QMIX+MARC, where MARC has a strong positive impact. VDN+MARC also shows improvements, although it stays below QMIX+MARC. Interestingly, when we look at other communication methods we can see that, in general, TARMAC [5] or MASIA [28] do not perform well in some tasks. We also note the inconsistent performance of COMMNET, which performs reasonably in some cases, but fails in others (Fig. 3(c)).

Taking a deeper look at the behaviours of the agents, the positive impact of MARC is further illustrated in Fig. 5 for $2 \mathrm { c } _ { - } \mathrm { v s } _ { - } 6 4 \mathrm { z } .$ . We can see in this figure that agents that use MARC to communicate (bottom) tend to remain cooperating close to each other for the entire episode, while when they do not communicate (top), they eventually start moving away from each other. The behaviours when using QMIX+COMMNET (middle) support our previous observations, where agents tend to be together more often when they communicate, as with MARC. However, with QMIX+COMMNET, the agents assume a less optimal strategy where they spent long periods of time in the middle of the enemy team, losing health points and resulting in the elimination of one of the agents, leaving the other alone to finish the game. This suboptimal behaviour does not happen with MARC and this is reflected in the team performances in Fig. 3.

To further demonstrate the strength of the proposed method, we evaluate MARC in a PredatorPrey environment [13]. Previous works such as [1], [36], [45] have demonstrated the importance of considering these scenarios that impose stronger punishments for non-cooperative behaviours. We consider a version of this environment where 4 agents must catch 2 moving prey in a $7 \times 7$ grid. At least two agents are needed to catch one prey, and when they do, the team receives a reward of $5 \times N$ , where N is the number of agents. For each step there is a small penalty of $- 0 . 0 1 \times N$ and, most importantly, there is a team penalty of $p \times N$ that punishes the agents when one of them attempts to catch a prey alone. In Fig. 4 we can see the rewards achieved by the experimented methods for different values of $p .$ When we use MARC, the agents take advantage of the messages sent by the others and can solve the task. While for $p = - 0 . 5 0$ communication does not seem to be critical, as we increase to $p = - 0 . 7 5$ (Fig. 4(b)) and $p ~ = ~ - 1 . 0$ (Fig. 4(c)), we can see that communication is necessary, and the agents cannot solve the task without it. Interestingly, while TARMAC and MASIA only managed to achieve positive rewards for $p = - 0 . 5 0$ (Fig. 4(a)), when we scale to $p = - 0 . 7 5$ and $p = - 1 . 0$ , MARC and COMMNET are the only ones to be successful. Overall, we observe the inconsistency of methods like COMMNET and TARMAC, which do well in some cases, but completely fail in others. On the other hand, MARC shows consistently good performances in all the experimented scenarios.

![](images/ae3aab700cd586318187ecd71b5cdeb4d3c71f78f998341620fe5cc20c9e3bb0.jpg)  
Fig. 5. Learned behaviours for QMIX (top), QMIX+COMMNET (middle), and QMIX+MARC (bottom) after training in 2c vs 64zg. When we use MARC, we can see that the agents will be always cooperating close to each other, while without communication they eventually move away from one another (top right). Additionally, we can see that with QMIX+COMMNET the agents communicate but learn a sub-optimal policy that results into one of them being eliminated (middle right).

## C. Performance in Other Complex Scenarios

To support the results presented in the previous subsection V-B, we demonstrate now the strength of the proposed method in other relevant complex environments. We analyse our method in Lumberjacks and TrafficJunction (environments as illustrated in Fig. 1(a) and 1(b)). As discussed in the previous subsection, in some SMAC environments communication might not be as important as in other scenarios. Thus, in addition to the PredatorPrey results in Fig. 4, we carried additional experiments in other complex environments. However, we still stress the importance of these methods being able to solve a wide range of environments, and not only scenarios where communication is needed. In order to get a better understading of these additional environments, we present below a brief description for each one of them.

Lumberjacks (Fig. 1(a)) consists of an environment where 4 agents must chop all the existing 12 trees in a 8 × 8 map. In this environment, the agents receive a penalty of −1 every timestep and a reward of +10 when a tree is cut [13]. Each tree is assigned a random level between 1 and the number of agents, where this level represents the number of agents required at the same time to cut the tree. Importantly, the high step penalty and the elevated need for cooperation make this task challenging.

TrafficJunction (Fig. 1(b)) represents a cross-shaped traffic junction where agents coming from 4 different entries in the junction must cross towards a pre-defined location at the end of another road after crossing the junction [13]. In our case, we have defined the number of agents to 10 agents, making this task very challenging. The agents received a penalty of −0.01 to incentivize them to keep moving and a penalty of −10 if they collide with another agent.

In the results depicted in Fig. 6(a) and Fig. 6(b) we observe that, in these complex environments, communication shows to have a strong impact and the proposed method demonstrates substantially improved performances over the baselines. This kind of environments has been used in other works to evaluate the strength of communication, such as in [6]. Importantly, we note once again the robustness of MARC across different environments, while others might work well in some of them, but perform poorly in others (as discussed in the previous subsection V-B).

## D. Additional Results: Message Compression and Representation

1) Compressing Messages: In the previous section, the results presented assume optimal conditions where communication among entities can be done in a lossless manner, i.e., where messages can be exchanged without loss of information. In this subsection, we present the results of communicating under a lossy communication channel, using the same proposed architecture. More specifically, we use the DCT to compress the messages into a smaller representation. Importantly, this does not affect the size of the communication endpoints, since the messages are reconstructed once they reach the destination, and thus there is no need to modify the size of the networks for different levels of compression.

![](images/981d87288e9ab7e9a7ee738551b3165592e4660d31d3f34ed2cb9faf7f694c89.jpg)  
(a) Lumberjacks

![](images/7816107da49154e38d5391e59930eeac8c78bebe643c1d6891c569e153cc261e.jpg)  
(b) Traffic Junction

Fig. 6. Additional results in (a) Lumberjacks and (b) Traffic Junction. The plots (a) and (b) depict the average team reward over time.  
![](images/87d793428ca7ad96e0c6cec82605dac7a2e4f68dbc3b2262536210519c95bf35.jpg)  
(a) 3s vs 5z

![](images/1a494a4e662cdce0a3cc9c489b9f8edd7893b91fdeb17e23ccc12ebe1e655b93.jpg)  
(b) 2c vs 64zg

Fig. 7. Results for VDN+MARC when we use the DCT to compress the messages that the agents generate. To evaluate the impact of the lossy compression of the DCT, we compress the messages to 20%, 40%, 60%, and 80% of their original size, as illustrated in the figure.  
![](images/184c4b84d3b0cdad567097198ae56573ca44aa0946d3fa507c54b37e76854985.jpg)  
(a) 3s vs 5z

![](images/983a6f5688df6499949ddae825acecda26b3430a7b00ed859f80b907f1578044.jpg)  
(b) 2c vs 64zg  
Fig. 8. Results for QMIX+MARC when we use the DCT to compress the messages that the agents generate. To evaluate the impact of the lossy compression of the DCT, we compress the messages to 20%, 40%, 60%, and 80% of their original size, as illustrated in the figure.

Fig. 7 illustrates the performances of VDN+MARC under different levels of DCT lossy message compression. In the figure, we can see that VDN+MARC can still learn even when using compressed messages, despite there is also a decrease of performance in both of the attempted scenarios. However, in Fig. 7(b), there is still a big increase of performance when compared to VDN without communication in this environment (as seen before in Fig. 3). Importantly, this means that, even when using compressed and hence lightweight messages, the performance can still be improved with communication when it is necessary.

![](images/c935b64d90f5224f6c687e568d4be2801114406563b58187f156b3cbda79dec4.jpg)  
Fig. 9. TSNE plot of the messages learned by each agent in a Switch task, after training using VDN+MARC. Each number in the plot represents the respective time step in the environment until termination, next to the message generated by each agent at that timestep. Note that, while the depicted episode lasted for 13 timesteps, above we show the states for the timesteps from 2 to 7 only.

In addition to the effects of compression on VDN+MARC, we extend our experiments to evaluate the effect of compression also on QMIX+MARC. Fig. 8(a) depicts the performances of QMIX+MARC under different levels of DCT lossy message compression. As expected, we can see that the performances will decrease as the level of message compression increases. However, it is important to note that the agents can still learn the tasks despite the high levels of compression. In some cases where communication channels might be constrained to a certain bandwidth, it is important to ensure that communication can still be leveraged to improve cooperation in MARL.

To investigate how the compression method affects the messages learned by the agents, we have looked into the messages devised by the agents for a 40% level of compression with QMIX+MARC in the 2c vs 64zg environment. In Fig. 10 we can see that the messages will occupy smaller ranges (mainly evident when closer to the end of the episode), than when compared to when there is no compression (which will be discussed in the next subsection). This suggests that the agents try to find more specific messages with less broad values when these must be compressed afterwards.

2) Messages Learned: One of the motivations for the MARC architecture is to enable the agents to produce more meaningful messages of their perceptions of the environment. To further support our motivations, we have trained VDN+MARC in the Switch environment from [13] where two agents need to reach their destination by crossing a corridor where only one of them fits at a time. Fig. 9 shows the TSNE plot of the messages learned by the agents on a successful episode after training the team in this environment (note that the episode lasts for 13 timesteps but only the steps from 2 to 7 are illustrated). In the plot we can clearly see that, at each timestep, each agent learns a distinct message that contains a meaning that is attached to that particular observation. At the time of negotiating their passage $( t = [ 3 , \ldots , 5 ] )$ , in the screenshots above the plot we can see that one agent waits for the other to pass and only after will start to cross, while taking into account the messages received.

![](images/b9d3d67f2fcba57f9f369e42cd4b341c1401f29623a2f446baf0c9ff0a6954c7.jpg)  
(a) QMIX+MARC with 40% DCT compression

![](images/f92706c02abc4728feedc6b7d9b887ad79e169c3710def9b72e6447b8e7d7489.jpg)  
(b) QMIX+MARC with 40% DCT compression

![](images/c19ca768c2606813a7a85246ee9a1f48d49e003f830055449f42d2fe29381567.jpg)  
(a) No regularizer.H = 2.065.

![](images/8f529249b8f752710c54b9a29c7e76872731b97301bc5843b0b7c526ed41230b.jpg)  
Fig. 10. Messages generated by the two agents over the course of a successful $2 \mathrm { c } _ { - } \mathrm { v s } _ { - }$ 64zg episode, after being trained with QMIX+MARC under a message compression level of 40% with the DCT (messages plotted after being decompressed at the destination using the IDCT).

## VI. CONCLUSION AND FUTURE WORK

## E. Ablations: Message Regularizer

To analyse the messages learned, we have saved the trained networks of QMIX+MARC for the 2c vs 64zg task, both using and not using the regularizer (purple box in Fig.2). Fig. 11 shows the messages produced by the two agents involved in 2c vs 64zg during a successful episode, i.e., an episode where the agents win the game. We can see in Fig. 11(c) and 11(d) that the messages produced when the agents use the message regularizer are compact within a smaller interval (around [−4, 4]) when compared to the messages produced when we do not use the regularizer (Fig. 11(a) and 11(b)). In the latter case, the messages produced assume values that lie inside a much larger range (around [−10, 10]), meaning that the possible kind of messages learned by the agents is more ambiguous, and making the task more difficult for them since the messages will not be as precise and meaningful as in the case of the message regularizer. The values of the differential entropy H for the messages produced that are shown in the caption of the figures also support our observations that messages that were produced with the help of the regularizer will contain less uncertainty and ambiguity. These observations support the results presented in subsubsection V-D2 showcasing messages other messages produced through MARC.

We have previously described how MARC uses a message regularizer to improve the quality of the messages learned during training. In this subsection, we analyse the impact of the message regularizer on the messages produced in the learning problem.

Communication frameworks in Multi-Agent Reinforcement Learning (MARL) represent a transformative approach to developing cooperative strategies within decentralized systems. By enabling agents to synthesize and share local observations, communication facilitates a higher level of behavioral synchronization and predictive coordination. In this work, we introduced Multi-Agent Regularized Communication (MARC), a modular framework compatible with diverse value-function factorization methods. Our experimental results demonstrate that MARC significantly enhances performance in complex tasks by fostering the emergence of highly representative communication protocols through its novel information-inspired regularization mechanism.

(b) No regularizer.H = 2.172.  
![](images/166099a45800a598f850de209f14cf17f6ad07deb2b4ce194d2f929511fa3c70.jpg)

![](images/b506b790c267c429f30ea8b428f781cc78605442cad466bc1ecd45bacc074b90.jpg)  
(c) With regularizer.H = 1.279.  
(d) With regularizer.H = 1.293.  
Fig. 11. Messages learned by the two agents in 2c vs 64zg throughout one successful episode with QMIX+MARC when the message regularizer is removed (a-b), and when it is used (c-d). The values of H denote the differential entropy of the messages.

Furthermore, we investigated a critical gap between MARL theory and the physical constraints of real-world deployment: limited communication bandwidth. By integrating the Discrete Cosine Transform (DCT) for message compression, we demonstrated that robust distributed intelligence can be maintained even under significant information loss. This finding validates that communication-based cooperation is viable for resource-constrained hardware and high-latency environments where spectral efficiency is paramount.

While MARC provides a robust foundation, several avenues for future research remain. We aim to investigate the impact of communication across varying network capacities and heterogeneous agent architectures. Additionally, while the current work operates within the CTDE paradigm, future iterations will explore fully decentralized training schemes to further relax the reliance on centralized mixers. Finally, we intend to study how adaptive message compression affects learning across different network scales and extend these findings to practical robotic testbeds to showcase the effects of communicating in dynamic, real-world environments.

## REFERENCES

[1] Z. Liu, L. Wan, X. sui, K. Sun, and X. Lan, “Multi-Agent Intention Sharing via Leader-Follower Forest,” arXiv, Tech. Rep. arXiv:2112.01078, Dec. 2021, arXiv:2112.01078 [cs] type: article. [Online]. Available: http://arxiv.org/abs/2112.01078

[2] J. N. Foerster, Y. M. Assael, N. de Freitas, and S. Whiteson, “Learning to Communicate with Deep Multi-Agent Reinforcement Learning,” in Advances in Neural Information Processing Systems, vol. 29, May 2016, arXiv: 1605.06676.

[3] S. Yanes Luis, D. Shutin, J. Marchal Gomez, D. Guti´ errez Reina, and´ S. Toral Mar´ın, “Deep reinforcement multiagent learning framework for information gathering with local gaussian processes for water monitoring,” Advanced Intelligent Systems, vol. 6, no. 8, p. 2300850, 2024. [Online]. Available: https://advanced.onlinelibrary.wiley.com/doi/ abs/10.1002/aisy.202300850

[4] W. Kim, J. Park, and Y. Sung, “Communication in Multi-Agent Reinforcement Learning: Intention Sharing,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=qpsl2dR9twy

[5] A. Das, T. Gervet, J. Romoff, D. Batra, D. Parikh, M. Rabbat, and J. Pineau, “TarMAC: Targeted Multi-Agent Communication,” in Proceedings of the 36th International Conference on Machine Learning, vol. 97, Jul. 2019, pp. 1538–1546, arXiv: 1810.11187.

[6] S. Sukhbaatar, a. szlam, and R. Fergus, “Learning Multiagent Communication with Backpropagation,” in Proceedings of the 30th International Conference on Neural Information Processing Systems, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, Eds., 2016, pp. 2252–2260.

[7] F. A. Oliehoek, M. T. J. Spaan, and N. Vlassis, “Optimal and Approximate Q-Value Functions for Decentralized POMDPs,” J. Artif. Int. Res., vol. 32, no. 1, pp. 289–353, May 2008, place: El Segundo, CA, USA Publisher: AI Access Foundation.

[8] L. Kraemer and B. Banerjee, “Multi-agent reinforcement learning as a rehearsal for decentralized planning,” Neurocomputing, vol. 190, pp. 82–94, May 2016. [Online]. Available: https://linkinghub.elsevier.com/ retrieve/pii/S0925231216000783

[9] L. Canese, G. C. Cardarilli, L. Di Nunzio, R. Fazzolari, D. Giardino, M. Re, and S. Spano, “Multi-Agent Reinforcement Learning:\` A Review of Challenges and Applications,” Applied Sciences, vol. 11, no. 11, p. 4948, May 2021. [Online]. Available: https: //www.mdpi.com/2076-3417/11/11/4948

[10] Z. Cheng, D. Ye, T. Zhu, W. Zhou, P. S. Yu, and C. Zhu, “Multi-agent reinforcement learning via knowledge transfer with differentially private noise,” International Journal of Intelligent Systems, vol. 37, no. 1, pp. 799–828, 2022. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/int.22648

[11] Y. Du, B. Liu, V. Moens, Z. Liu, Z. Ren, J. Wang, X. Chen, and H. Zhang, “Learning correlated communication topology in multi-agent reinforcement learning,” in Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems, ser. AAMAS ’21. Richland, SC: International Foundation for Autonomous Agents and Multiagent Systems, 2021, p. 456–464.

[12] X. Wang, X. Li, J. Shao, and J. Zhang, “Ac2c: Adaptively controlled two-hop communication for multi-agent reinforcement learning,” 2023.

[13] A. Koul, “ma-gym: Collection of multi-agent environments based on openai gym.” https://github.com/koulanurag/ma-gym, 2019.

[14] C. Resnick, A. Gupta, J. Foerster, A. M. Dai, and K. Cho, “Capacity, Bandwidth, and Compositionality in Emergent Language Learning,” arXiv:1910.11424 [cs, stat], Apr. 2020, arXiv: 1910.11424. [Online]. Available: http://arxiv.org/abs/1910.11424

[15] R. Wang, X. He, R. Yu, W. Qiu, B. An, and Z. Rabinovich, “Learning efficient multi-agent communication: An information bottleneck approach,” CoRR, vol. abs/1911.06992, 2019. [Online]. Available: http://arxiv.org/abs/1911.06992

[16] K. S. Mohamed, Wireless Communication Systems: Compression and Decompression Algorithms. Cham: Springer International Publishing, 2022, pp. 27–53. [Online]. Available: https://doi.org/10.1007/978-3-031-19297-5 2

[17] J. Wang, Z. Ren, T. Liu, Y. Yu, and C. Zhang, “Qplex: Duplex dueling multi-agent q-learning,” 2021.

[18] P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls, and T. Graepel, “Value-decomposition networks for cooperative multi-agent learning,” 2017.

[19] T. Hu, B. Luo, C. Yang, and T. Huang, “Mo-mix: Multi-objective multi-agent cooperative decision-making with deep reinforcement learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 12 098–12 112, 2023.

[20] S. Liu, J. Song, Y. Zhou, N. Yu, K. Chen, Z. Feng, and M. Song, “Interaction pattern disentangling for multi-agent reinforcement learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–15, 2024.

[21] T. Rashid, M. Samvelyan, C. S. de Witt, G. Farquhar, J. Foerster, and S. Whiteson, “QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning,” in Proceedings of the 35th International Conference on Machine Learning, vol. 80, Jul. 2018, pp. 4295–4304, arXiv: 1803.11485.

[22] S. SHEN, M. Qiu, J. Liu, W. Liu, Y. Fu, X. Liu, and C. Wang, “Resq: A residual q function-based approach for multi-agent reinforcement learning value factorization,” in Thirty-Sixth Conference on Neural Information Processing Systems, 2022. [Online]. Available: https: //openreview.net/forum?id=bdnZ 1qHLCW

[23] M. Zhou, Y. Chen, Y. Wen, Y. Yang, Y. Su, W. Zhang, D. Zhang, and J. Wang, “Factorized q-learning for large-scale multi-agent systems,” in Proceedings of the First International Conference on Distributed Artificial Intelligence, ser. DAI ’19. New York, NY, USA: Association for Computing Machinery, 2019. [Online]. Available: https://doi.org/10.1145/3356464.3357707

[24] J. Jiang and Z. Lu, “Learning Attentional Communication for Multi-Agent Cooperation,” arXiv:1805.07733 [cs], Nov. 2018, arXiv: 1805.07733. [Online]. Available: http://arxiv.org/abs/1805.07733

[25] S. Q. Zhang, Q. Zhang, and J. Lin, “Efficient communication in multi-agent reinforcement learning via variance based control,” in Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and´ R. Garnett, Eds., vol. 32. Curran Associates, Inc., 2019. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2019/file/14cfdb59b5bda1fc245aadae15b1984a-Paper.pdf

[26] ——, “Succinct and robust multi-agent communication with temporal message control,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 17 271– 17 282. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2020/file/c82b013313066e0702d58dc70db033ca-Paper.pdf

[27] T. Chu, S. Chinchali, and S. Katti, “MULTI-AGENT REINFORCE-MENT LEARNING FOR NETWORKED SYSTEM CONTROL,” p. 17, 2020.

[28] C. Guan, F. Chen, L. Yuan, C. Wang, H. Yin, Z. Zhang, and Y. Yu, “Efficient multi-agent communication via self-supervised information aggregation,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 1020–1033. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2022/file/075b2875e2b671ddd74aeec0ac9f0357-Paper-Conference.pdf

[29] I. Mordatch and P. Abbeel, “Emergence of Grounded Compositional Language in Multi-Agent Populations,” arXiv, Tech. Rep. arXiv:1703.04908, Jul. 2018, arXiv:1703.04908 [cs] type: article. [Online]. Available: http://arxiv.org/abs/1703.04908

[30] S. Gupta, R. Hazra, and A. Dukkipati, “Networked Multi-Agent Reinforcement Learning with Emergent Communication,” arXiv:2004.02780 [cs], Apr. 2020, arXiv: 2004.02780. [Online]. Available: http://arxiv.org/abs/2004.02780

[31] I. Kajic, E. Aygun, and D. Precup, “Learning to cooperate: Emergent¨ communication in multi-agent navigation,” CoRR, vol. abs/2004.01097, 2020. [Online]. Available: https://arxiv.org/abs/2004.01097

[32] T. Lin, J. Huh, C. Stauffer, S. N. Lim, and P. Isola, “Learning to ground multi-agent communication with autoencoders,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., vol. 34. Curran Associates, Inc., 2021, pp. 15 230– 15 242. [Online]. Available: https://proceedings.neurips.cc/paper files/ paper/2021/file/80fee67c8a4c4989bf8a580b4bbb0cd2-Paper.pdf

[33] D. Kim, S. Moon, D. Hostallero, W. J. Kang, T. Lee, K. Son, and Y. Yi, “Learning to schedule communication in multi-agent reinforcement learning,” arXiv preprint arXiv:1902.01554, 2019.

[34] F. A. Oliehoek and C. Amato, A Concise Introduction to Decentralized POMDPs. Springer, 2016.

[35] W. Li, X. Wang, B. Jin, D. Luo, and H. Zha, “Structured cooperative reinforcement learning with time-varying composite action space,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8618–8634, 2022.

[36] K. Son, D. Kim, W. J. Kang, D. Hostallero, and Y. Yi, “QTRAN: learning to factorize with transformation for cooperative multi-agent reinforcement learning,” CoRR, vol. abs/1905.05408, 2019. [Online]. Available: http://arxiv.org/abs/1905.05408

[37] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. A. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nature, vol. 518, pp. 529–533, 2015. [Online]. Available: https://api.semanticscholar.org/CorpusID: 205242740

[38] N. Ahmed, T. Natarajan, and K. Rao, “Discrete cosine transform,” IEEE Transactions on Computers, vol. C-23, no. 1, pp. 90–93, 1974.

[39] J. Makhoul, “A fast cosine transform in one and two dimensions,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 1, pp. 27–34, 1980.

[40] J. K. Gupta, M. Egorov, and M. Kochenderfer, “Cooperative multi-agent control using deep reinforcement learning,” in Autonomous Agents and

Multiagent Systems, G. Sukthankar and J. A. Rodriguez-Aguilar, Eds. Cham: Springer International Publishing, 2017, pp. 66–83.

[41] T. Jaakkola, M. Jordan, and S. Singh, “Convergence of stochastic iterative dynamic programming algorithms,” in Advances in Neural Information Processing Systems, J. Cowan, G. Tesauro, and J. Alspector, Eds., vol. 6. Morgan-Kaufmann, 1993. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 1993/file/5807a685d1a9ab3b599035bc566ce2b9-Paper.pdf

[42] F. Melo, “Convergence of q-learning: a simple proof,” pp. 1–4, 2001.

[43] M. Samvelyan, T. Rashid, C. S. de Witt, G. Farquhar, N. Nardelli, T. G. J. Rudner, C. Hung, P. H. S. Torr, J. N. Foerster, and S. Whiteson, “The starcraft multi-agent challenge,” CoRR, vol. abs/1902.04043, 2019. [Online]. Available: http://arxiv.org/abs/1902.04043

[44] T. Wang, J. Wang, C. Zheng, and C. Zhang, “Learning nearly decomposable value functions via communication minimization,” CoRR, vol. abs/1910.05366, 2019. [Online]. Available: http://arxiv.org/ abs/1910.05366

[45] W. Bohmer, V. Kurin, and S. Whiteson, “Deep coordination¨ graphs,” CoRR, vol. abs/1910.00091, 2019. [Online]. Available: http://arxiv.org/abs/1910.00091