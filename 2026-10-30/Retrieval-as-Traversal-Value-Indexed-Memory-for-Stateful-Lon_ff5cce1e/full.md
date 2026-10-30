# Retrieval as Traversal: Value-Indexed Memory for Stateful Long-Context Agents

Daniel Strauß Technical University of Munich

Zhenshan Bing Nanjing University

Alois Knoll Technical University of Munich

## Abstract

Long-context models repeatedly solve closely related retrieval problems. Attention or global retrieval can retain large stores, but each forward pass again searches for what matters. Recurrent compression avoids this search only by deciding earlier what information to preserve. Neither mechanism directly exploits that useful memories at neighbouring retrieval steps may substantially overlap. We propose an alternative in which retrieval is a persistent local traversal through an address space. We separate a memory’s content from its address. Utility is defined by the information revealed by retrieving the content, while geometry determines only how cheaply that information can be reached. This leads to a cost-regularised indexing objective that preserves information value inside local neighbourhoods while charging for every memory made locally accessible. Its marginal-value form gives both attractive and exclusion pressures. Thus, complementary memories should be co-retrievable when the additional value exceeds retrieval cost, whereas redundant memories should not occupy the same retrieval set solely because they are individually useful. As a proof of principle, we test only the simplest fixedgeometry instance on MiniGrid Memory. A persistent local traversal head reaches 0.95 mean return after 0.59M training steps with a five-token context, compared with 0.94M for Gaussian absolute addressing and more than 2M for categorical absolute addressing. The experiment does not test learned geometry or repeated retrieval. Instead, it motivates the hypothesis that stateful retrieval may be easier to learn than stateless. To test the full proposed method further experimentation is necessary.

## 1 Introduction

Long-context models face a basic trade-off. Keeping a large history available preserves information, but attention or retrieval must repeatedly determine which parts of that history matter. Compressing the past into a bounded recurrent state avoids the scan, but the compression is performed before all future information needs are known. Sparse attention, approximate nearest-neighbour search, and external retrieval reduce the cost of a query, but typically still treat successive queries as fresh matching problems [5, 7, 9, 6].

This misses repetition, a third source of cost. The memories useful at two nearby inference steps need not be independent sets. In order to avoid this cost information about value of the to be retrieved memories has to be known before retrieving them. We investigate this idea as retrieval by traversal. Memories and the current context and retrieval heads are placed in an address space. The positions of the current context and the retrieval heads aim to hold the information about the current expected value of the to be retrieved memories, as all memories in ϵ distance will be retrieved.

Our proposal differs from semantic retrieval in one important respect: addresses should be shaped by the decision value of retrieving memory contents, rather than by similarity between the contents themselves. This distinction matters because two semantically similar memories may be redundant, whereas two dissimilar memories may only be useful together.

This paper makes two deliberately limited contributions. First, we give a decision-theoretic objective for learning such a geometry and derive how redundancy and complementarity affect local co-retrieval. Second, we report a small proof-of-principle experiment in which the geometry is fixed and only the persistent retrieval mechanism is learned. We do not claim that the learned value-index proposed here has yet been experimentally validated.

Related work. External differentiable memories have long separated computation from stored state. Neural Turing Machines combine content and location addressing [3]. Differentiable Neural Computers add dynamic allocation and temporal links [4]. Lie-access Neural Turing Machines are particularly close in mechanism. Here, memories occupy a continuous address manifold and read heads move by relative transformations [12]. Neural Map instead uses physical position as an external memory index [8]. Modern sparse-attention and retrieval systems use hashing, learned routing, nearest-neighbor search, or retrieval indices to avoid attending to every stored item [5, 9, 11, 6]. We ask instead whether the address space itself can be organized by the expected utility of future retrieval and then traversed persistently across related queries.

## 2 Value-indexed memory

## 2.1 Content, context, and address

The content of a memory must be distinguished from the location at which it is stored. Let

$$
\Omega _ { t } = \{ 1 , \ldots , n _ { t } \}
$$

index the external memories available at time t. Memory j contains a random variable

$$
M _ { j } \in { \mathcal { M } } .
$$

Before retrieval, $M _ { j }$ is unobserved from the model’s current information state and retrieving the memory reveals a realized content $m _ { j }$

Let $C _ { j } ^ { \mathrm { w } }$ denote the context in which memory $j$ was written. Its address is

$$
z _ { j } = \Phi ( C _ { j } ^ { \mathrm { w } } ) \in \mathbb { R } ^ { R } .\tag{1}
$$

Conditional on a realized memory store, $z _ { j }$ is a fixed location. Retrieving $j$ inserts $m _ { j }$ , not $C _ { j } ^ { \mathrm { w } }$ , into the model’s working context.

The current context $C _ { t }$ , consisting of recent observations together with recently retrieved memory, contains the sequence of information currently available to the predictor. The position of the retrieval head is maintained separately and is not part of $C _ { t }$

For a realized current context c and a set $S \subseteq \Omega _ { t }$ , let $M _ { S }$ denote the corresponding memory contents in a fixed canonical order. We define their pre-query information value by

$$
V ( S ; c ) = \mathbb { E } [ U ( c \oplus M _ { S } ) - U ( c ) \mid C _ { t } = c ] ,\tag{2}
$$

where U is task utility, such as expected return or negative predictive loss, and ⊕ denotes incorporation into the working context. V may also be learned from the next token predictor, through counterfactual training.

Importantly, V is defined from memory contents and no address appears in (2). Geometry affects which contents are cheaply retrievable, but not what those contents are worth.

For the theoretical discussion we assume free disposal, which is that additional information can not reduce the expected utility. Hence

$$
S \subseteq T \quad \Longrightarrow \quad V ( S ; c ) \leq V ( T ; c ) .\tag{3}
$$

## 2.2 A cost-regularised local index

Let

$$
B _ { \varepsilon } ( x ) = \{ j \in \Omega _ { t } : \| z _ { j } - x \| \leq \varepsilon \}
$$

denote the memories locally accessible from address x. A context c is mapped to a query position $\Phi ( c )$ , giving

$$
N _ { c } ( \Phi ) = B _ { \varepsilon } ( \Phi ( c ) ) .\tag{4}
$$

A naive objective would minimise only the information missing from this neighbourhood,

$$
V ( \Omega _ { t } ; c ) - V ( N _ { c } ( \Phi ) ; c ) .
$$

Under (3), however, this objective has a trivial solution: mapping every context and memory to the same point puts all of $\Omega _ { t }$ inside every ball and gives zero information loss.

We therefore charge a retrieval cost $\lambda > 0$ per accessible memory:

$$
\boxed { \mathcal { L } _ { \lambda } ( \Phi ) = \mathbb { E } [ V ( \Omega _ { t } ; C _ { t } ) - V ( N _ { C _ { t } } ( \Phi ) ; C _ { t } ) + \lambda | N _ { C _ { t } } ( \Phi ) | ] . }\tag{5}
$$

The outer expectation ranges over contexts and realised memory stores. The constant embedding now pays $\lambda | \Omega _ { t } |$ at every query and is no longer universally optimal.

For a fixed context, minimising (5) is equivalent to selecting a locally accessible set with large

$$
\begin{array} { r } { J _ { \lambda } ( S ; c ) = V ( S ; c ) - \lambda | S | . } \end{array}\tag{6}
$$

Thus λ replaces an arbitrary top-k budget by a value threshold: another memory is worth making accessible precisely when the information it adds justifies its retrieval cost.

## 2.3 Complementarity and redundancy induce opposite pressures

Define the marginal value of memory j after already having access to $S$ as

$$
\Delta _ { j } ( S ; c ) = V ( S \cup \{ j \} ; c ) - V ( S ; c ) .\tag{7}
$$

For two memories $i , j \notin S$ , define their conditional interaction

$$
\begin{array} { c } { { s _ { i j } ( S ; c ) = V ( S \cup \{ i , j \} ; c ) - V ( S \cup \{ i \} ; c ) } } \\ { { - V ( S \cup \{ j \} ; c ) + V ( S ; c ) } } \\ { { = \Delta _ { j } ( S \cup \{ i \} ; c ) - \Delta _ { j } ( S ; c ) . } } \end{array}\tag{8}
$$

Positive $s _ { i j }$ is complementarity, meaning that observing i increases the marginal value of j. Negative $s _ { i j }$ is redundancy.

Observation 1 (Cost-adjusted co-retrieval). For any context c, background set $S ,$ and $i , j \notin S ,$

$$
\begin{array} { r l } { } & { J _ { \lambda } ( S \cup \{ i , j \} ; c ) - J _ { \lambda } ( S \cup \{ i \} ; c ) } \\ { } & { = \Delta _ { j } ( S \cup \{ i \} ; c ) - \lambda . } \end{array}\tag{9}
$$

Hence joint retrieval is preferred to retrieving i alone exactly when $\Delta _ { j } ( S \cup \{ i \} ; c ) > \lambda .$

Proof. Substitute (6) and cancel common terms.

The observation gives the geometry both an attractive and an exclusion criterion. If i makes $j$ sufficiently valuable that

$$
\Delta _ { j } ( S \cup \{ i \} ; c ) > \lambda ,
$$

then a local neighbourhood benefits from making the pair jointly accessible. Positive interaction $s _ { i j }$ increases the contexts in which this occurs.

Conversely, suppose $j$ is redundant once i is known. In the exact case,

$$
V ( S \cup \{ i , j \} ; c ) = V ( S \cup \{ i \} ; c ) ,
$$

so $\Delta _ { i } ( S \cup \{ i \} ; c ) = 0$ and retrieving both loses exactly λ relative to retrieving only i. The index therefore has no incentive to place both in every query neighborhood based on their high singleton values alone. Due to this exclusion pressure, a learned geometry should make redundant alternatives separately reachable when possible. In particular, $\lVert z _ { i } - \bar { z } _ { j } \rVert > \dot { 2 } \varepsilon$ is sufficient to ensure that no single ε-ball contains both, although the objective does not require this specific geometric solution.

This argument concerns decision-theoretic redundancy, instead of raw statistical correlation between $M _ { i }$ and $M _ { j }$ . Covariance of hidden representations is representation-dependent and need not indicate whether two memories provide the same information for the downstream decision.

Substitutes compete for the same coalitions. The interaction structure also formalises a stronger notion of redundancy. Call memories i and $j$ value-equivalent substitutes if, for every context c and every background set S not containing them,

$$
V ( S \cup \{ i \} ; c ) = V ( S \cup \{ j \} ; c ) = V ( S \cup \{ i , j \} ; c ) .\tag{10}
$$

Then

$$
\Delta _ { i } ( S ; c ) = \Delta _ { j } ( S ; c )
$$

for every $S , c .$ For another coalition A, define the change in i’s marginal value caused by A,

$$
\eta _ { i } ( A \mid S , c ) = \Delta _ { i } ( S \cup A ; c ) - \Delta _ { i } ( S ; c ) .\tag{11}
$$

Equation (10) immediately implies

$$
\boxed { \eta _ { i } ( A \mid S , c ) = \eta _ { j } ( A \mid S , c ) \qquad \mathrm { f o r ~ a l l } \ A , S , c . }\tag{12}
$$

Thus exact substitutes have the same individual value and additionally share same interaction profile with every other coalition. In this sense, redundant memories compete for the same synergistic sets.

This suggests that the learned address geometry should be driven by value and interaction profiles, rather than by similarity of the raw memory contents.

## 2.4 Persistent retrieval as traversal

Once a useful geometry exists, retrieval need not restart globally at every step. Let $i _ { t } \in \mathbb { R } ^ { R }$ be a persistent retrieval-head position and define

$$
M _ { t } = B _ { \varepsilon } ( \Phi ( c _ { t } ) ) \cup B _ { \varepsilon } ( i _ { t } ) , \qquad W _ { t } = \bigcup _ { \tau = t - H + 1 } ^ { t } M _ { \tau } .\tag{13}
$$

The working window therefore retains recently traversed memories, while the head continues to move locally. The movement rule is left modular. It may be learned from RL as in our experiment, moving to the memory of highest value in the last forwards pass, or by extending the existing key, value estimators. It is future work to find the movement rule which is most canonical given the construction of Φ.

The regime in which persistence can help can be described without choosing a fixed k. Define the cost-optimal retrieval set

$$
S _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { S \subseteq \Omega _ { t } } \left[ V ( S ; c _ { t } ) - \lambda | S | \right]\tag{14}
$$

and its one-step addition drift

$$
D _ { t } = | S _ { t } ^ { \star } \rangle S _ { t - 1 } ^ { \star } | .\tag{15}
$$

When $D _ { t }$ is small relative to $| S _ { t } ^ { \star } |$ , most useful information persists between consecutive retrievals. A persistent mechanism can in principle reuse the previous retrieval state instead of rediscovering the full set.

## 3 A highly crude fixed-Φ proof of principle

Learning Φ is the central unresolved part of the proposal. We therefore test a deliberately weaker question. Given that the correct address space is supplied by the environment, is then relative and persistent retrieval easier to learn than selecting absolute memory locations?

A deliberately crude approximation to Φ. The theoretical model learns $z _ { j } = \Phi ( C _ { i } ^ { \mathrm { w } } )$ from value relations. The present experiment does not. Instead, it replaces Φ by a hand-designed map derived from the spatiotemporal state at which the memory was created, schematically

$$
\Phi _ { 0 } ( C _ { j } ^ { \mathrm { w } } ) = ( x _ { j } , y _ { j } , t _ { j } ) .\tag{16}
$$

Thus the experiment assumes that physical proximity and temporal proximity are useful indexing features.

Task and architecture. We use MiniGrid Memory [1]. The agent observes a cue, traverses a corridor, and later chooses the corresponding object at a T-junction. The policy uses a Transformer-XL backbone [2] and PPO [10]. At each step the action is

$$
a _ { t } = ( a _ { t } ^ { \mathrm { e x t } } , a _ { t } ^ { \mathrm { m e m } } ) ,
$$

where the first component acts in the environment and the second moves the retrieval head.

The memory address space is fixed to the environment’s spatial and temporal coordinates. In this early stage experiment no value-indexed embedding is learned. Retrieved observations and current observations enter a shared FIFO context window. Models are first pretrained without a retrieval head on the fixed-size task with context length 64, then fine-tuned on the randomised variant with context length 5. In the randomised task, the relevant cue lies between 7 and 17 environment steps before the decision.

We compare relative head movement with two heads that select an absolute memory position, parameterised respectively by a Gaussian and by a categorical distribution. Six seeds are used for each retrieval-head condition and three for the reduced-context no-head baseline.

Results. The primary learning-time statistic is the training step at which the mean learning curve across seeds first exceeds 0.95 return. We did not choose the mean of per-seed threshold-crossing times, since some individual runs never cross the threshold within the training budget.

Table 1: Comparison of retrieval heads on MiniGrid-Memory-S17Random (6 seeds each). Models were pretrained on MiniGrid-Memory-S9 without a retrieval head using a context size of 64. "Steps to 0.95" is the first training step at which the mean (min–max) return reaches 0.95. "> 2.0" indicates that the value was not determined after 2,048,000 steps. The entry No retrieval head (baseline) was tested with only 3 seeds.
<table><tr><td>Method</td><td>Steps to 0.95 (M)</td><td>Avg. Excess Steps</td><td>Context</td></tr><tr><td>No retrieval head (as pretrained)</td><td>0.14 (0.12–0.16)</td><td>5.08 (4.80 – 6.23)</td><td>64</td></tr><tr><td>No retrieval head (baseline)</td><td>1.38 (1.29 – 1.43)</td><td>15.0 (9.16 – 20.73)</td><td>5</td></tr><tr><td>Memory Navigation</td><td>0.59 (0.52 – 0.65)</td><td>6.18 (5.95 – 6.33)</td><td>5</td></tr><tr><td>Absolute Position – Gaussian</td><td>0.94 (0.82 − &gt;2.0)</td><td>6.32 (6.09 – 6.58)</td><td>5</td></tr><tr><td>Absolute Position – Categorical</td><td>&gt;2.0 (0.63 – &gt;2.0)</td><td>7.42 (5.81 – 9.40)</td><td>5</td></tr></table>

Relative movement reaches the 0.95 threshold after 0.59M steps, compared with 0.94M for Gaussian absolute addressing and more than 2.0M for categorical absolute addressing. It also produces trajectories close to the full-context reference despite receiving only five context tokens.

This result supports only the preliminary claim, that incremental relative addressing is easier to learn than the tested absolute-addressing alternatives in this single-retrieval task. It does not test the learned indexing objective (5), interaction-driven geometry, or the low-drift hypothesis.

A revealing negative control. The reduced-context baseline is stronger than expected. With a short window, the agent can encode the cue indirectly into its own behaviour. With temporal encoding available, it waits for a cue-dependent number of steps and later reads the resulting clock state. When temporal and spatial encodings are removed and the window is shortened further, it can instead repeatedly turn in the corridor and encode the cue in its recent trajectory. These are legitimate solutions. They prevent us from claiming a capability gap and make the current experiment evidence only for learning and behavioural efficiency.

## 4 Limitations

The strongest claims of the framework remain untested. The experiment uses an environmentprovided geometry rather than learning Φ from (5). Additionally, the experiment requires the retrieval of only one piece of information and therefore does not test whether the proposed method helps with sequentially accessing memories when they become useful.

It has also not yet been explored how well a fixed-dimensional geometry can represent the retrieval sets favored by ${ \mathcal { L } } _ { \lambda } ( \Phi )$ . If two memories $M _ { i }$ and $M _ { j }$ are quite similar, they may be redundant in many contexts, i.e. $s _ { i j } ( c ) < 0$ , while at the same time being complementary in many other contexts, i.e. $s _ { i j } ( c ) > 0$ . Can the optimal Φ position $z _ { i }$ and $z _ { j }$ such that they are contained in the same ϵ-ball primarily in contexts in which they are complementary, or is such a spatial arrangement impossible?

Additionally, the retrieval rule is left modular rather than canonical. A slight modification to the proposed method is needed so that the movement of the retrieval head and the construction of Φ arise from the same underlying principle.

Finally, whilst the loss term is given, it is not described how Φ should be trained in clear detail. Exploring the behavior of $s _ { i j }$ with respect to Φ may result in a class of additional constraints such that the loss can be decomposed into multiple terms that are easier to train. The training procedure itself is also not discussed.

## 5 Conclusion

We propose treating long-context retrieval as persistent traversal through an address space learned from retrieval utility. Separating memory content from memory address yields a simple organizing principle. Geometry should make valuable combinations locally accessible while charging for unnecessary co-retrieval. The resulting objective naturally distinguishes complementarity from redundancy and shows that exact substitutes share the same interaction profile with other memory coalitions. Our current MiniGrid experiment tests only the traversal mechanism under fixed geometry. The next empirical tests would be learning the value-indexed space and evaluating repeated low-drift retrieval.

## References

[1] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and J K Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. Advances in neural information processing systems, 36:73383–73394, 2023.

[2] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings of the 57th annual meeting ofthe associationfor computational linguistics, pages 2978–2988, 2019.

[3] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines, 2014. URL https: //arxiv.org/abs/1410.5401.

[4] Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou,´ et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538 (7626):471–476, 2016.

[5] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.

[6] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[7] Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836, 2018.

[8] Emilio Parisotto and Ruslan Salakhutdinov. Neural map: Structured memory for deep reinforcement learning. arXiv preprint arXiv:1702.08360, 2017.

[9] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

[10] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[11] Yuhuai Wu, Markus N Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. arXiv preprint arXiv:2203.08913, 2022.

[12] Greg Yang and Alexander M. Rush. Lie-access neural turing machines. CoRR, abs/1611.02854, 2016. URL http://arxiv.org/abs/1611.02854.

## A Open access to data and code

The experimental code is available at https://anonymous.4open.science/r/ Retrieval-as-Traversal-008D. A script for reproducing the results reported in this paper is included. The experiments require less than 3 GB of VRAM, and a single training run takes about three hours on a consumer-grade desktop computer.

## B Declaration of LLM Usage

General-purpose LLMs, such as ChatGPT and Claude, were used to assist the development of the experimental code. They were also used as interactive research assistants during the development of the theoretical formulation, including to critique mathematical constructions, find counterexamples and suggest possible refinements. Additionally, they assisted writing and editing. All resulting definitions, arguments, derivations, and claims included in the paper were reviewed and verified by the authors, who take full responsibility for the content.