# Transformers Provably Learn Graph Search: Training Dynamics and the Exponential Power of Depth

Xutao Ma Somayeh Sojoudi

UC Berkeley

{xutao\_ma,sojoudi}@berkeley.edu

November 11, 2026

## Abstract

Large Language Models (LLMs) have achieved remarkable progress on complex reasoning tasks, yet the mechanisms by which they acquire reasoning capabilities and the eficiency of their reasoning remain poorly understood. In this paper, we investigate the path-finding problem over directed graphs—a symbolic abstraction of multi-step reasoning. We first characterize the architectural requirements for this task, proving that one-layer transformers require $\Omega ( N ^ { 2 } )$ size on dense graphs to implement the key DFS child-selection primitive. While a two-layer transformer implements full DFS with O(N log(N)) size. Thus, a second layer is necessary and suficient for near-linear DFS reasoning. We then extend this expressivity analysis to deeper models: for every $k \geq 3$ , a k-layer transformer implements a bounded-reachability DFS policy with look-ahead horizon $1 + 2 ^ { k - 3 }$ , establishing an exponential increase in reachability horizon with depth. Finally, going beyond expressivity, given appropriately curated training data, we show that two-layer transformers can provably learn to execute DFS via gradient flow and generalize to unseen graphs. Together, our results provide a mechanistic account of how transformers implement graph search, how depth expands bounded look-ahead, and how such reasoning algorithms can emerge through training. We provide the experiment code and Lean 4 proofs of all the theoretical results.<sup>1</sup>

## 1 Introduction

Large Language Models (LLMs) have demonstrated striking reasoning capabilities across mathematics, code, scientific question answering, and planning, especially when equipped with chain-of-thought (CoT) prompting [Wei et al., 2022] and test-time scaling of intermediate reasoning steps [Guo et al., 2025]. Yet our theoretical understanding of these capabilities remains incomplete. We still lack precise answers to several basic questions: how can a transformer implement a multi-step reasoning algorithm, why does architectural depth improve reasoning eficiency, and under what conditions can such an algorithm be acquired through gradient-based training?

A growing body of theoretical work studies these questions through symbolic reasoning tasks, where the algorithmic structure is explicit enough to analyze. Prior results have clarified important aspects of transformer expressivity, showing both limitations of constant-depth models on inherently sequential tasks and the additional power obtained from chain-of-thought computation [Feng et al., 2023, Li et al., 2024, Merrill and Sabharwal, 2024]. Other work has examined depth–width tradeofs for graph algorithms [Sanford et al., 2024a, Yehudai et al., 2025], sample complexity for reasoning tasks [Wies et al., 2023], and training dynamics for algorithmic primitives such as induction heads, sparse token selection, in-context regression, and Markov-chain structure learning [Tian et al., 2023, Nichani et al., 2024, Chen et al., 2024, Wang et al., 2024].

Despite this progress, graph search remains poorly understood. This setting is especially important because it captures a central dificulty of reasoning: the model must not merely follow a single predetermined chain, but must choose among alternatives, remember what has already been explored, and backtrack when a branch fails. In other words, graph search formalizes the kind of committed exploration and revision that appears in planning, proof search, and multi-step deduction.

Existing theory leaves three gaps. First, most provable learning results for path-finding focus on trees, where there is at most one path between two vertices and the need for genuine search is largely absent [Yang et al., 2025]. Directed graphs are qualitatively harder: they contain branching, cycles, and dead ends, so a successful algorithm must maintain a visited set and decide when to move forward or backtrack. Second, although prior work shows that constant-depth transformers can implement DFS [Merrill and Sabharwal, 2024], the architectural threshold for eficient graph search is not known. It is unclear whether one layer sufices, why two layers should be special, and how additional layers translate into improved search eficiency. Third, recent work has analyzed reachability with continuous chain of thought [Zhu et al., 2025a], but the corresponding learnability question for standard discrete CoT traces remains open.

In this paper, we give a theoretical account of how transformers implement, scale, and learn depth-first search (DFS) on directed graphs. We focus on DFS because it is the canonical sequential search algorithm: it combines branching, memory of visited states, forward exploration, and backtracking, while naturally producing a discrete CoT trace. Our results connect three aspects of reasoning—architecture, depth, and training dynamics—within a single graph-search framework.

## Our contributions.

• We identify a sharp depth threshold for eficient DFS. We prove that a one-layer transformer cannot implement even a basic DFS child-selection primitive on dense directed graphs without quadratic representation size, measured as heads × width × precision. In contrast, we construct a two-layer transformer that implements full DFS with near-linear representation size. The separation concerns the width a single attention layer needs, not parameter count or running time, and it shows that the second layer is not merely a convenience: it is necessary and suficient for exact DFS with sub-quadratic width in our setting.

• We show that depth gives an exponential increase in look-ahead horizon. Beyond the two-layer DFS construction, we prove that additional layers can perform parallel boundedreachability computation. After explicit neighbor aggregation, each propagation layer doubles the radius represented in source and child memories. For every $k \geq 3 ,$ a k-layer transformer implements a bounded-reachability DFS policy with horizon $1 + 2 ^ { k - 3 }$ . This establishes a depth–horizon tradeof. We distinguish it from worst-case search-eficiency guarantees, which require additional control over the distances of the selected branches.

• We prove that DFS behavior can emerge through gradient-based training. For a two-layer transformer trained on a permutation-symmetric DFS curriculum, we analyze the gradient-flow dynamics of the relevant attention parameters. The dynamics preserve a lowdimensional invariant structure, and the resulting model generalizes from the training curriculum to unseen graphs and DFS traces. Thus, the construction is not only expressive: it can be learned under a mathematically analyzable training process.

Together, these results provide a mechanistic account of transformer-based graph search. They show that eficient DFS requires a genuine two-layer mechanism, that additional depth yields exponentially larger look-ahead, and that the corresponding search behavior can arise from gradientbased training on discrete chain-of-thought traces. More broadly, our analysis uses directed-graph path finding as a clean theoretical proxy for understanding how reasoning capabilities depend on architecture, depth, and training data.

## 2 Preliminaries

Notation. For $N \in \mathbb { N } .$ , let $[ N ] = \{ 1 , \dots , N \}$ . We use A to denote the token vocabulary, reserving V for the vertex set of a graph. For a finite set K, let $S _ { K }$ denote the set of all permutations on $\kappa$ We write $\tau _ { \mathrm { P r o m p t } }$ for a prompt, viewed as a sequence of tokens. Throughout the paper, I denotes the identity matrix, and E denotes an all-ones matrix of the appropriate size.

Directed graphs and permutations. A directed graph of order N is an ordered pair $\mathcal { G } = ( \nu , \mathcal { E } )$ 2 where $\nu = [ N ]$ is the vertex set and $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ is the edge set. $\mathrm { A }$ path from a source node $n _ { s }$ to a target node $n _ { t }$ is a sequence of vertices

$$
p = ( n _ { 1 } , n _ { 2 } , . . . , n _ { k } )
$$

with $n _ { 1 } = n _ { s }$ and $n _ { k } = n _ { t }$ , such that $( n _ { i } \to n _ { i + 1 } ) \in \mathcal { E }$ for every $i \in [ k - 1 ]$ . A shortest path is any such path with the minimum number of vertices. For any permutation π of $\nu ,$ , we define the permuted graph by

$$
\pi ( { \mathcal { G } } ) = ( \mathcal { V } , \pi ( { \mathcal { E } } ) ) , \qquad \pi ( { \mathcal { E } } ) = \{ ( \pi ( u ) \to \pi ( v ) ) : ( u \to v ) \in { \mathcal { E } } \} ,
$$

and define the permuted path as $\pi ( p ) = ( \pi ( n _ { 1 } ) , \ldots , \pi ( n _ { k } ) )$

Path-finding problem and depth-first search. Given a directed graph ${ \mathcal { G } } ,$ a source node $n _ { s }$ , and a target node $n _ { t }$ , the path-finding problem asks for a path from $n _ { s }$ to $n _ { t } .$ , if one exists. Depth-first search (DFS) is a recursive search procedure for this problem. Starting from $n _ { s } .$ , DFS repeatedly explores outgoing edges from the current node n. If there exists an unvisited child v with $( n  v ) \in \mathcal { E }$ , the search moves to v, marks it as visited to avoid cycles, and continues recursively. If no unvisited child remains, the search backtracks to the parent of n in the current search stack and explores alternative branches. This forward-and-backtracking process terminates either when $n _ { t }$ is found, yielding a valid $n _ { s }  n _ { t }$ path, or when the search returns to $n _ { s }$ after exhausting all reachable branches.

Multi-head transformer architecture. An L-layer transformer with H attention heads is a sequence-to-sequence model. Given an input sequence $\pmb { X } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { T } ] \in \mathbb { R } ^ { d \times T }$ , the model computes hidden states $\pmb { X } ^ { ( l ) } \in \mathbb { R } ^ { d \times T }$ as follows. Let $\pmb { X } ^ { ( 1 ) } = \pmb { X }$ , and for every $l \in [ L ]$

$$
\pmb { X } ^ { ( l + 1 ) } = \pmb { X } ^ { ( l ) } + \sigma \Bigl ( \mathrm { A t t e n } ^ { ( l ) } ( \pmb { X } ^ { ( l ) } ) \Bigr ) ,\tag{1}
$$

$$
\mathrm { A t t e n } ^ { ( l ) } ( { \pmb X } ^ { ( l ) } ) = { \pmb W } _ { O } ^ { ( l ) } \left[ \begin{array} { c } { { \mathrm { H e a d } _ { 1 } ^ { ( l ) } ( { \pmb X } ^ { ( l ) } ) } } \\ { { \vdots } } \\ { { \mathrm { H e a d } _ { H } ^ { ( l ) } ( { \pmb X } ^ { ( l ) } ) } } \end{array} \right] ,\tag{2}
$$

$$
\mathrm { H e a d } _ { h } ^ { ( l ) } ( { \pmb X } ^ { ( l ) } ) = { \pmb W } _ { V } ^ { ( l , h ) } { \pmb X } ^ { ( l ) } \phi ^ { ( l ) } \Bigl ( ( { \pmb X } ^ { ( l ) } ) ^ { T } ( { \pmb W } _ { K } ^ { ( l , h ) } ) ^ { T } { \pmb W } _ { Q } ^ { ( l , h ) } { \pmb X } ^ { ( l ) } \Bigr ) .\tag{3}
$$

Here $\sigma$ is an element-wise activation function; throughout, we take $\sigma ( x ) = \operatorname { R e L U } ( x ) = \operatorname* { m a x } \{ 0 , x \}$ The map $\mathrm { A t t e n } ^ { ( l ) }$ denotes the attention block at layer l, ${ W } _ { O } ^ { ( l ) }$ is the output projection, $\boldsymbol { W } _ { V } ^ { ( l , h ) }$ ${ \pmb W } _ { K } ^ { ( l , h ) }$ , and ${ W } _ { Q } ^ { ( l , h ) }$ are the value, key, and query matrices for head $h ,$ , and $\phi ^ { ( l ) }$ is an element-wise attention activation, possibly with normalization.

The transformer output logits are

$$
\mathrm { T F } ( \boldsymbol { X } ; \theta ) = \boldsymbol { W } _ { R } ^ { T } \boldsymbol { X } ^ { ( L + 1 ) } ,\tag{4}
$$

where $\theta$ denotes the trainable parameters and $W _ { R } \in \mathbb { R } ^ { d \times | \mathcal { A } | }$ is the readout matrix. The next-token distribution is the softmax over these logits:

$$
p _ { \theta } ( v ; \boldsymbol { X } ) = \frac { \exp ( \mathrm { T F } ( \boldsymbol { X } ; \theta ) _ { v } ) } { \sum _ { v ^ { \prime } \in \mathcal { A } } \exp ( \mathrm { T F } ( \boldsymbol { X } ; \theta ) _ { v ^ { \prime } } ) } ,\tag{5}
$$

where $\operatorname { T F } ( X ; \theta ) _ { \mathit { \iota } }$ is the logit assigned to token v.

Prompt format and chain of thought. Following Zhu et al. [2025b,a], the transformer pathfinding prompt has three parts: (1) a beginning-of-sequence token $< \mathsf { s } > ; ~ ( 2 )$ a graph-description segment containing M edges, where each edge is represented by a source token $s _ { i } .$ , a target token $t _ { i } ,$ and a special edge token $< \mathsf { e } > ;$ and (3) a task-description segment containing the target token $n _ { \mathrm { T a r } }$ a start-reasoning token ${ < } r >$ , and the root token $n _ { \mathrm { R o o t } }$ . We denote the full prompt by $\tau _ { \mathrm { P r o m p t } } , i . e .$

$$
\tau _ { \mathrm { P r o m p t } } = < 5 > , s _ { 1 } , t _ { 1 } , < 6 > , \cdots , s _ { M } , t _ { M } , < 6 > , n _ { \mathrm { T a r } } , < \mathsf { r } > , n _ { \mathrm { R o o t } } .\tag{6}
$$

Starting from $n _ { \mathrm { R o o t } }$ , the transformer predicts tokens autoregressively, generating a chain-of-thought (CoT) trace $n _ { 1 } , n _ { 2 } , \ldots , n _ { T }$ . Generation terminates when the model reaches the target $n _ { \mathrm { T a r } } ,$ or when it returns to the root $n _ { \mathrm { R o o t } }$ after all of the root’s child nodes have been explored.

## 3 Expressivity

Although Merrill and Sabharwal [2024] showed that constant-depth transformers can perform DFS, the minimal architectural requirements remain unclear. In this section, we first prove that any onelayer transformer that performs DFS on directed graphs of order N must have $\Omega ( N ^ { 2 } )$ representation size, i.e., heads × width × precision, in the dense-graph regime. This lower bound shows that at least two layers are needed for DFS with sub-quadratic representation size. We then give a two-layer construction with near-linear representation size, and finally show that additional layers provide an exponential improvement in look-ahead depth.

## 3.1 One-layer lower bound for DFS child selection

To formalize the size measure, we first specify the numerical precision of a transformer.

Definition 3.1 (p-bit precision). A transformer uses p bits of precision if all coordinates produced by $w _ { K } , W _ { Q } , W _ { V } , W _ { O }$ , and the attention heads, after rounding, can be represented with p bits without changing the computed input-output function. As in Sanford et al. [2024b], attention sums are accumulated in $p \textmd { - }$ -bit fixed-point arithmetic: for an input of length $T ,$ , every unnormalized attention weight $w _ { j }$ and every summand $w _ { j } W _ { V } { \pmb x } _ { j }$ of an attention numerator is rounded to a p-bit fixed-point number, and $p \geq \log _ { 2 } T$ . Consequently, the partial sum of an attention numerator or normalizer over any subset of positions is exactly representable with $p + \lceil \log _ { 2 } T \rceil = O ( p )$ bits per coordinate. The value dimension of every head is at most d.

Size measure. As in Sanford et al. [2024b], we quantify a transformer layer by $H d p ,$ , the number of heads times the embedding dimension times the precision, and call this its representation size; it is the quantity that Sanford et al. [2024b] call simply the size, and we use the longer name because it is a measure of width and precision rather than of parameters or running time. It is the number of bits that one layer of attention can deliver to a single position, which is exactly the quantity controlled by the communication-complexity argument below, and we claim no separation in parameter count or running time. Dense $d \times d$ projection matrices have $\Theta ( d ^ { 2 } ) = \Theta ( N ^ { 2 } )$ entries in the one-layer and in the two-layer setting alike (the matrices of our construction are built from identity and all-ones blocks, but they are still $N \times N )$ , and on a dense graph the prompt itself has length $\Theta ( N ^ { 2 } )$ , so any model processes $\Theta ( N ^ { 2 } )$ tokens. The separation below is therefore a statement about depth versus width: a single attention layer must route $\Omega ( N ^ { 2 } )$ bits to the output position, whereas with two layers O(N log N) bits per layer sufice, which is within a logarithmic factor of the $\Omega ( N )$ bits needed to represent the visited set at all (Section B.3).

We next define a restricted task that isolates the retrieval of an outgoing child.

Definition 3.2 (Unique outgoing-child task). Fix $N , M \in \mathbb { N }$ . Let $\mathcal { G } = ( [ N ] , \mathcal { E } )$ be a directed graph with exactly M distinct edges, $\mathcal { E } = \{ ( s _ { i } \to t _ { i } ) : i \in [ M ] \}$ . The prompt is $< \mathsf { s } > , s _ { 1 } , t _ { 1 } , < \mathsf { e } > , \hdots , s _ { M } , t _ { M } , < \mathsf { e } > , q _ { \hdots }$ an enumeration of E followed by a query node $q ,$ under the promise that exactly one edge has source $q .$ The desired output is the target of this unique outgoing edge.

In the hard instances constructed below the unique outgoing edge of $q$ is not a self-loop, so its target is the mandatory first DFS move from $q ,$ and the reduction applies to the full DFS next-step task as well.

Theorem 3.3 (One-layer representation-size lower bound). Let $M \geq 1$ and $N \geq 3 \lceil \sqrt { M } \rceil + 1$ Fix any scalar weighting function $\phi : \mathbb { R }  \mathbb { R }$ . Any one-layer transformer with the architecture of Section ${ \mathcal { Q } } ,$ with element-wise attention activation $\phi ,$ , either normalized or unnormalized, with H heads, embedding dimension $d ,$ and $p { - } b i t$ precision that is correct on every valid instance of the unique outgoing-child task whose graph has vertex set [N] and exactly M distinct edges satisfies $H d p = \Omega ( M )$

Proof. See Section B.1.

The proof adapts the communication-complexity argument for one-layer lower bounds on induction heads [Sanford et al., 2024b]: we reduce from INDEX [Kushilevitz, 1997] by placing the M records in distinct cells of an $m \times m$ grid, $m = \lceil \sqrt { M } \rceil$ , so that the source of a record encodes its row and the target encodes its column and Alice’s bit; Bob changes only the source of the selected record to the query vertex $q .$ This yields exactly M distinct edges on 3m + 1 vertices, and since Alice controls only the target-token positions while the final query is fixed, each head contributes an $O ( d p ) { \mathrm { - b i t } }$ summary to a one-way protocol.

Corollary 3.4 (Dense-graph lower bound). For $N \geq 4$ , set m $\mathbf { \Lambda } _ { N } = \lfloor ( N - 1 ) / 3 \rfloor$ and $M _ { N } = m _ { N } ^ { 2 }$ . Any one-layer transformer in the architecture of Theorem 3.3 that is correct on every valid instance of the unique outgoing-child task whose graph has vertex set [N] and exactly MN $M _ { N }$ distinct edges requires $H d p = \Omega ( N ^ { 2 } )$ . Consequently, any one-layer transformer in this architecture that solves the full DFS next-step task on all N-vertex directed graphs requires $\Omega ( N ^ { 2 } )$ representation size in the worst case, already on a family with $\Theta ( N ^ { 2 } )$ distinct edges.

Proof. See Section B.2.

## 3.2 Two-layer near-linear construction for exact DFS

Token embeddings and the first layer. Each token is embedded in four N-dimensional node blocks (plus a fixed positional component): edge endpoints $s _ { i } , t _ { i }$ and the target $n _ { \mathrm { T a r } }$ place their one-hot vector in the fourth block, the root and the CoT tokens place theirs in the first block, and the special tokens are zero (Section B.3). The O(N) dimension is within a logarithmic factor of what exact DFS requires, since the model must distinguish $2 ^ { N }$ visited sets (Section B.3). The first layer only copies information from the preceding token.

Lemma 3.5 (Predecessor-copy layer, informal version of Theorem B.1). Assume that the input embeddings are augmented with a sinusoidal positional encoding, or any positional encoding that permits a constant-size relative-position selector. Then, for every leakage parameter $\delta \in ( 0 , 1 / 8 )$ there exists a one-layer transformer block with O(1) attention heads, embedding dimension $O ( N )$ and parameter precision O(log N) that copies the first block of the preceding token into the second block and its fourth block into the third block, each one-hot vector scaled by a factor in $[ 1 - 2 \delta , 1 ]$ and all other copied coordinates exactly zero after the ReLU.

Concretely, up to the scalar factors in [1 − 2δ, 1] on the copied one-hot vectors, the hidden states entering the second layer are

$$
\mathrm { E d g e ~ s o u r c e ~ } s _ { i } : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { s _ { i } } \end{array} \right) , \qquad \mathrm { E d g e ~ t a r g e t ~ } t _ { i } : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { s _ { i } } \\ { t _ { i } } \end{array} \right) , \quad \begin{array} { l }  < s > : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) , \quad < \mathsf { e } > : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { t _ { i } } \\ { 0 _ { N } } \end{array} \right) , \end{array}\tag{7}
$$

$$
\begin{array} { r } { \mathrm { R o o t } n _ { \mathrm { R o o t } } : \left( \begin{array} { c } { n _ { \mathrm { R o o t } } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) , \qquad \mathrm { T a r g e t } n _ { \mathrm { T a r } } : \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { n _ { \mathrm { T a r } } } \end{array} \right) , \quad < \Gamma > : \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { n _ { \mathrm { T a r } } } \\ { 0 _ { N } } \end{array} \right) , \mathrm { C o T } \mathrm { s t e p } n _ { t } : \left( \begin{array} { c } { n _ { t } } \\ { n _ { t - 1 } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) , } \end{array}
$$

with $n _ { 0 } = n _ { \mathrm { R o o t } }$ . Every edge-target token thus exposes its edge $( s _ { i } \to t _ { i } )$ in the last two blocks, and every CoT token exposes the transition $n _ { t - 1 } \to n _ { t }$ in the first two.

Second layer. The second layer uses two heads with linear attention activation, $\phi ^ { ( 2 ) } ( x ) = x$ , and the weights

$$
\begin{array} { r } { W _ { K Q } ^ { ( 2 , 1 ) } = \left( \begin{array} { c c c c } { c _ { 1 } E } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { c _ { 1 } I } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , \quad W _ { K Q } ^ { ( 2 , 2 ) } = \left( \begin{array} { c c c c } { c _ { 2 } I } & { 0 } & { 0 } & { 0 } \\ { c _ { 2 } I } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { W _ { V } ^ { ( 2 , 1 ) } = \left( \begin{array} { l } { - I } \\ { 0 } \\ { 0 } \\ { I } \end{array} \right) ^ { T } , \quad W _ { V } ^ { ( 2 , 2 ) } = \left( \begin{array} { l } { - I } \\ { I } \\ { 0 } \\ { 0 } \end{array} \right) ^ { T } , \quad W _ { \mathcal { O } } ^ { ( 2 ) } = \left( \begin{array} { l l } { 0 } & { 0 } \\ { 0 } & { 0 } \\ { I } & { 0 } \\ { 0 } & { I } \end{array} \right) , \quad W _ { R } = \left( 0 \quad 0 \quad I \quad I \right) ^ { T } , } \end{array}
$$

where $W _ { K Q } ^ { ( l , h ) } = ( W _ { K } ^ { ( l , h ) } ) ^ { T } W _ { Q } ^ { ( l , h ) }$ is the aggregated key-query matrix, all matrices are displayed restricted to the four node blocks and vanish on the constant and positional coordinates, $W _ { R }$ has zero columns for the special tokens, and $c _ { 1 } > c _ { 2 } > 0$ are constants. The leakage parameter of the first layer is chosen as

$$
\delta \leq \frac { c _ { 1 } - c _ { 2 } } { 4 c _ { 1 } } ,\tag{9}
$$

so that the first-layer copies, whose scalar factors lie in $[ 1 - 2 \delta , 1 ]$ , cannot reorder the two scores below. The two heads implement the two DFS actions. The forward head matches, for the current node $n _ { t } ,$ the edge tokens with source $n _ { t }$ and writes their destinations into the readout block, while subtracting every node that already occurs in the CoT trace; an unvisited child thus receives score $\approx c _ { 1 }$ and a visited child is canceled. The backtracking head attends to the CoT transitions entering and leaving $n _ { t } ;$ completed excursions cancel and the tree-edge arrival leaves score $\approx c _ { 2 }$ on the stack parent. The backtracking candidate is therefore always available but is dominated whenever an unvisited child exists.

A CoT trace $\tau = ( n _ { 0 } , n _ { 1 } , \dots , n _ { t } )$ with $n _ { 0 } = n _ { \mathrm { R o o t } }$ is a valid DFS trace if every step follows the DFS rule of Section 2, and it is nonterminated if DFS has a valid next move from it, i.e., $n _ { t } \neq n _ { \mathrm { T a r } }$ and $n _ { t }$ either has an unvisited child or is not the root.

Theorem 3.6 (Two-layer transformers can perform DFS with near-linear representation size). Suppose the first layer is constructed as in Theorem B.1 with a leakage parameter $\delta$ satisfying Equation (9), and the second layer has weights Equation (8). Then the resulting two-layer transformer has $O ( 1 )$ heads per layer, embedding dimension $O ( N )$ , and ${ \cal O } ( \log N )$ -bit precision, hence representation size O(N log N), and for any graph $\mathcal { G } = ( \mathcal { V } = [ N ] , \mathcal { E } )$ and any valid nonterminated DFS trace $\tau$ on $\mathcal { G } _ { i }$ every token in $\begin{array} { r } { \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { \theta } ( v ; \tau _ { P r o m p t } , \tau ) } \end{array}$ is a valid DFS move.

Proof sketch. The representation-size bound is that of Theorem B.1, since the second-layer weights are constants. Fix a valid nonterminated trace with current node $n _ { t }$ . By Equation (7), the forward head assigns score $c _ { 1 } \lambda$ with $\lambda \in \left[ 1 - 2 \delta , 1 \right]$ to each edge-target token with source $n _ { t }$ and score exactly $c _ { 1 }$ to each CoT token, and its value map adds the destination of the former and subtracts the node of the latter; before the ReLU, an unvisited child of $n _ { t }$ therefore has forward score at least $c _ { 1 } ( 1 - 2 \delta )$ while a visited child and a non-child have score at most 0. The backtracking head attends to the CoT tokens that enter or leave n ; as $n _ { t }$ is on top of the stack, it has left only by forward moves, so each completed excursion to a child yields a departure and a return term that cancel up to the leakage (at most $2 \delta c _ { 2 }$ remains on that child), while the stack parent receives only the tree-edge arrival, between $c _ { 2 } ( 1 - 2 \delta )$ and $c _ { 2 }$ . After the ReLU and the readout, every unvisited child has logit at least $c _ { 1 } ( 1 - 2 \delta )$ , the stack parent has logit in $[ c _ { 2 } ( 1 - 2 \delta ) , c _ { 2 } ]$ , and every other vertex has logit at most $2 \delta c _ { 2 } ;$ by Equation (9) and $\delta < 1 / 4$ these three bands are ordered, so every maximizer is an unvisited child when one exists and the stack parent otherwise. Full details are in Section B.5.

## 3.3 Multi-layer look-ahead construction and exponential depth advantage

Additional layers can expand the bounded reachability information available before each DFS move: after an explicit initialization, each propagation layer doubles the radius represented on the existing edge records.

Definition 3.7 (L-look-ahead DFS). Let $L \geq 1$ be an integer and fix the outgoing-edge order by increasing destination label. Given a valid nonterminated DFS state with current node $u ,$ target $n _ { \mathrm { T a r } } ,$ and visited set $S ,$ first consider the unvisited children v with $( u \to v ) \in \mathcal { E }$ and $\mathrm { d i s t } _ { \mathcal { G } } ( v , n _ { \mathrm { T a r } } ) \leq L - 1$ If this set is nonempty, move to its first vertex in the fixed order. Otherwise, apply the ordinary DFS rule: move to the first unvisited child if one exists, and otherwise backtrack to the parent in the current DFS stack.

Here distance is measured in the original graph and a zero-length path is allowed; the rule tests a Boolean bounded-reachability predicate and does not require a qualifying child to minimize distance.

Theorem 3.8 (Exponential look-ahead depth advantage). For every integer $k \geq 3$ , there exists a k-layer transformer with unmasked attention, $O ( 1 )$ heads per layer, embedding dimension $O ( N )$ , and $O _ { k } ( \log ( N + 1 ) )$ -bit fixed-point precision, whose token embeddings are augmented with the positional encoding of Theorem B.1 together with a constant coordinate, a beginning-of-sequence indicator, and a period-three position indicator (Section B.6), that implements L-look-ahead DFS with $L = 1 + 2 ^ { k - 3 }$ Its representation size is $O _ { k } ( N \log ( N + 1 ) )$ , where the subscript indicates that the constant may depend on k.

Proof sketch. The first layer creates persistent source and destination records on the edge-target positions, copies the goal, and prepares the predecessor information used by ordinary $\mathrm { D F S }$ . In the second layer, two linear-attention heads aggregate outgoing neighbors, one querying with the source identifier of each edge record and one with its destination identifier, which initializes source and child reachability memories whose supports are the sets reachable within one step from u and v for the record $( u  v )$ . Each of the next $k - 3$ layers doubles both radii by querying source identifiers with the current memories; the memories are nonnegative counts and only their supports matter. The final layer combines the goal coordinate of the child memory, the outgoing adjacency of the current node, and the visited-node counts; two fresh ReLU blocks and a linear readout implement the eligibility test, followed by the fixed child order and the ordinary DFS fallback. The complete construction appears in Section B.6. □

Thus depth gives an exponential increase in the radius of the bounded-reachability test. This Boolean test alone does not imply shortest-path selection or an exponential improvement in worstcase search eficiency; in Section B.6 we state the eficiency bound separately for a stronger policy that selects a child of minimum residual distance.

## 4 Training Dynamics

In this section we consider the training dynamics of the two-layer model constructed above. Since the predecessor-copy first layer is well studied [Bietti et al., 2023, Sanford et al., 2024b, Zhu et al., 2025b], we focus on the second layer, whose two heads implement the forward and the backtracking move. The core questions are how each head learns to attend to the right tokens and how the competition between the two heads evolves during training. We therefore fix $W _ { O } , W _ { V }$ as in the construction and train the key-query matrices

$$
\begin{array} { r } { W _ { K Q } ^ { ( 2 , 1 ) } = \left( \begin{array} { c c c c } { A _ { 1 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { A _ { 2 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , \quad W _ { K Q } ^ { ( 2 , 2 ) } = \left( \begin{array} { c c c c } { B _ { 1 } } & { 0 } & { 0 } & { 0 } \\ { B _ { 2 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , } \end{array}\tag{10}
$$

where $A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 } \in \mathbb { R } ^ { N \times N }$ are trainable. When decoding we use the dynamic readout $W _ { R } ( n _ { t } ) =$ $\left( 0 \quad 0 \quad D _ { n _ { t } } \quad D _ { n _ { t } } \right) ^ { T }$ with $D _ { n _ { t } } = \mathrm { d i a g } ( I _ { n _ { t } - 1 } , 0 , I _ { N - n _ { t } } )$ , so that the logit of the current node $n _ { t }$ is 0 while $n _ { t }$ remains a softmax candidate with weight $e ^ { 0 } = 1$ . Throughout, $p _ { \theta }$ is the softmax over the N vertex tokens (the special tokens are never admissible moves), the same readout is used for training and for the generalization loss below, and, since the loss is only piecewise smooth, gradient flow is understood in the Filippov sense $( { \dot { \theta } } \in - \alpha \partial { \mathcal { L } } ( \theta )$ with Clarke’s generalized gradient; on a switching surface the velocity is a convex combination of the one-sided fields).

Training data. All training data share a common graph skeleton $\mathcal { G } _ { 0 } = ( [ N ] , \mathcal { E } _ { 0 } )$ with $N \geq 7 .$

$$
\mathcal { E } _ { 0 } = \{ ( 1  i ) : 2 \leq i \leq K + 1 \} \cup \{ ( N - 1  N - 2 ) , ( N - 2  N - 3 ) , ( N - 3  1 ) \} ,\tag{11}
$$

where $2 \leq K \leq N { - } 5$ , so that the children $2 , \ldots , K { + } 1$ of node 1 are distinct from $N { - } 3 , N { - } 2 , N { - } 1 , N$ The canonical target and root are $( n _ { \mathrm { T a r } } , n _ { \mathrm { R o o t } } ) = ( N , N - 1 )$ , and the canonical CoT trajectory is

$$
\begin{array} { r l } & { \tau = \underbrace { N - 2  N - 3 } _ { \mathrm { p r e f i x ~ } \tau _ { \mathrm { p r e } } } \underbrace {  1  2  1  3  1  \dots  K + 1  1  N - 3 } _ { \mathrm { t r a i n i n g ~ t r a c e ~ } \tau _ { 0 } } } \\ & { \quad  N - 2  N - 1 . } \end{array}\tag{12}
$$

We train only on the $2 K + 2$ steps of $\tau _ { 0 }$ , which traverse all nodes reachable from $N - 3$ , visiting the children $2 , \ldots , K + 1$ of node 1 in turn and backtracking after each; training on the full trace leads to the same mechanism, but $\tau _ { 0 }$ keeps the gradient expressions clean and isolates the coupling between the two heads that drives the asymptotics.

Training loss. We use the cross-entropy of the next token: for the canonical trajectory, $\ell _ { \tau _ { \mathrm { P r o m p t } } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } } =$ $- \textstyle \sum _ { t = 1 } ^ { 2 K + 2 } \log p _ { \theta } ( \tau _ { 0 } [ t ] ; \tau _ { \mathrm { P r o m p t } } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } [ : t - 1 ] )$ , and the dataset loss is the permutation average

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathbb { E } _ { \pi \sim \mathrm { U n i f } ( S _ { \mathcal { V } } ) } \ell _ { \pi ( \tau _ { \mathrm { P r o m p t } } ) , \pi ( \tau _ { \mathrm { p r e } } ) , \pi ( \tau _ { 0 } ) } . } \end{array}\tag{13}
$$

$\mathcal { L } _ { \mathrm { t r a i n } }$ cannot reach 0, since diferent exploration branches conflict whenever $n _ { t }$ has several unexplored children, and it only measures performance on the training skeleton.

Generalization loss. At step t of a CoT trace $\tau$ we score all admissible DFS moves at once, $\begin{array} { r } { - \log \sum _ { v \in C _ { t } ( \tau ) } p _ { \theta } \big ( v ; \tau _ { \mathrm { P r o m p t } } , \tau [ : ~ t - 1 ] \big ) } \end{array}$ , where $C _ { t }$ is the set of unexplored children if nonempty and otherwise the singleton containing the stack parent of $n _ { t } ;$ this loss can be driven to 0. The generalization loss is the supremum over all directed graphs on $N$ vertices and all valid nonterminated DFS traces,

$$
\mathcal { L } _ { \mathrm { g e n } } = \operatorname* { s u p } _ { \varepsilon , \tau } \ - \ \sum _ { t = 1 } ^ { \mathrm { l e n } ( \tau ) } \log \Big ( \sum _ { v \in C _ { t } ( \tau ) } p _ { \theta } \big ( v ; \tau _ { \mathrm { P r o m p t } } , \tau [ : t - 1 ] \big ) \Big ) .\tag{14}
$$

Since the training data is fully symmetric, the permutation-invariant parameters form an invariant set of the gradient flow.

Proposition 4.1. Let $\mathcal { F }$ denote the set of parameters of the form

$$
\begin{array} { r } { A _ { 1 } = a _ { 1 } I + b _ { 1 } E _ { o f f } , \quad A _ { 2 } = a _ { 2 } I + b _ { 2 } E _ { o f f } , \quad B _ { 1 } = g _ { 1 } I + d _ { 1 } E _ { o f f } , \quad B _ { 2 } = g _ { 2 } I + d _ { 2 } E _ { o f f } , } \end{array}\tag{15}
$$

where $E _ { o f f } = E - I$ has all of-diagonal entries equal to 1 and zero diagonal. Then $\mathcal { F }$ is invariant under the gradient flow of Equation (13): the generalized gradient of $\mathcal { L } _ { t r a i n }$ lies in $\mathcal { F }$ at every point of ${ \mathcal F } ,$ and the flow restricted to $\mathcal { F }$ is a solution of the full flow, unique whenever the loss is diferentiable along it. In particular, matrices initialized in $\mathcal { F }$ remain of the form Equation (15) for all $t \geq 0$ . Moreover, on $\mathcal { F }$ the averaged loss equals the loss of any single permuted instance, $\mathcal { L } _ { t r a i n } ( \theta ) = \ell _ { \pi ( \tau _ { P r o m p t } ) , \pi ( \tau _ { p r e } ) , \pi ( \tau _ { 0 } ) } ( \theta )$ for every $\pi \in S _ { \mathcal { V } }$

Proof. See Section C.1.

We initialize in ${ \mathcal { F } } _ { : }$ with the of-diagonal entries of $A _ { 2 }$ and $B _ { 1 }$ strictly below their diagonal entries: for constants $0 < c _ { 0 } ^ { \prime } < c _ { 0 }$

$$
A _ { 1 } = c _ { 0 } E , \qquad A _ { 2 } = c _ { 0 } I + c _ { 0 } ^ { \prime } E _ { \mathrm { o f f } } , \qquad B _ { 1 } = c _ { 0 } I + c _ { 0 } ^ { \prime } E _ { \mathrm { o f f } } , \qquad B _ { 2 } = c _ { 0 } E ,\tag{16}
$$

that is, $a _ { 1 } = b _ { 1 } = a _ { 2 } = g _ { 1 } = g _ { 2 } = d _ { 2 } = c _ { 0 }$ and $b _ { 2 } = d _ { 1 } = c _ { 0 } ^ { \prime }$ . The strict separation $b _ { 2 } ( 0 ) < b _ { 1 } ( 0 )$ $g _ { 1 } ( 0 ) > d _ { 1 } ( 0 )$ places the initial point of the two ReLU switching surfaces $b _ { 2 } = b _ { 1 }$ and $g _ { 1 } = d _ { 1 }$ on the side that the dynamics then preserve (Theorem C.2). The all-constant initialization $c _ { 0 } \mathbf { { \cal E } }$ lies on these surfaces, where the vector field is discontinuous, and its trajectory depends on the convention for $\mathrm { R e L U ^ { \prime } ( 0 ) }$ and on $( N , K )$ : with $\mathrm { R e L U ^ { \prime } } ( 0 ) = 0$ the backtracking head never activates in exact arithmetic, and when N is large relative to K the trajectory first leaves towards $b _ { 2 } > b _ { 1 }$ (Section C.2).

Theorem 4.2. Let $A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 }$ be initialized as in Equation (16) with $0 < c _ { 0 } ^ { \prime } < c _ { 0 } \le c _ { * } N ^ { - 2 }$ where $c _ { * } = c _ { * } ( K ) > 0$ is the constant of Theorem C.2. Then, under gradient flow on Equation (13) with learning rate $\alpha$ , the scalar parameters of Equation (15) satisfy, as $t \to \infty$

$$
a _ { 2 } , b _ { 1 } = 2 \log ( \alpha t ) + O ( 1 ) , \qquad g _ { 1 } - d _ { 1 } = \log ( \alpha t ) + O ( 1 ) , \qquad | g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } | \le C ,\tag{17}
$$

where $C$ and the $O ( 1 )$ terms depend only on $N , K , c _ { 0 }$ , and $c _ { 0 } ^ { \prime }$ . As a consequence, there exist $t _ { * }$ and $C _ { N , K }$ , depending only on $N , K , c _ { 0 } , c _ { 0 } ^ { \prime } ,$ such that for all $t \ge t _ { * } / \alpha$

$$
\mathcal { L } _ { g e n } \leq C _ { N , K } \left( \alpha t \right) ^ { - \frac { N - 1 } { N } } .\tag{18}
$$

The exponent $( N - 1 ) / N$ in Equation (18) is sharp as $t \to \infty$ . Forward moves and backtracking moves to a non-root parent have admissible-token loss $O ( 1 / ( \alpha t ) )$ , with constants as large as $e ^ { a _ { 2 } - b _ { 1 } }$ , so these terms dominate ${ \mathcal { L } } _ { \mathrm { g e n } }$ over long horizons in practice (Section 5). When the parent of the current node is the root, however, its logit is $g _ { 1 } - d _ { 1 } + d _ { 2 }$ (Table 2), and $\begin{array} { r } { d _ { 2 } = - \frac { 1 } { N } \log ( \alpha t ) + O ( 1 ) } \end{array}$ because $\tau _ { 0 }$ never backtracks to the root, so $d _ { 2 }$ is driven only by the conserved combination $g _ { 2 } + ( N - 1 ) d _ { 2 } ;$ such a step costs $( \alpha t ) ^ { - ( N - 1 ) / N }$

Proof sketch. By Theorem 4.1, training the matrices is equivalent to training the eight scalars of Equation (15), whose gradients on the canonical data are explicit (Theorem C.2). The flow never moves $a _ { 1 }$ , and within an initial phase of length $\begin{array} { r } { T _ { 0 } = \frac { N } { 4 ( K + 1 ) \alpha } } \end{array}$ it drives b down to 0 and $d _ { 2 }$ to nonpositive values; from then on $b _ { 2 } \equiv 0$ , since its gradient vanishes on $\{ b _ { 2 } < 0 \}$ so that $b _ { 2 }$ cannot become negative, and the trajectory stays in the region $\{ b _ { 1 } > 0 , \ b _ { 2 } < b _ { 1 } , \ a _ { 2 } \geq b _ { 1 } , \ g _ { 1 } > d _ { 1 } , \ g _ { 2 } \geq d _ { 2 } \}$ . Writing $x _ { 1 } = a _ { 2 } , x _ { 2 } = b _ { 1 } , y _ { 1 } = g _ { 1 } - d _ { 1 } , y _ { 2 } = g _ { 2 } - d _ { 2 }$ and $u = x _ { 1 } - x _ { 2 } , v = y _ { 1 } - y _ { 2 }$ , the remaining dynamics form a closed four-variable system (Theorem C.3) in which the ReLU enters only through the indicators $\mathbb { 1 } _ { u \geq 0 }$ and $\mathbb { 1 } _ { v \geq 0 } ;$ the vector field is therefore discontinuous on the switching surfaces $u = 0$ and $v = 0$ , and the trajectory can slide along them (it does so along $v = 0$ , see Section 5). The main steps (Theorem C.5) are: (i) the sliding variable $v _ { + }$ stays bounded, which we show with a Lyapunov function of the form $W = Z s ^ { p }$ built from the softmax normalizers; (ii) the gaps u and $2 y _ { 1 } - x _ { 1 }$ are bounded above by barrier arguments; (iii) with the gaps controlled, the leading terms of the equations for $X = x _ { 1 }$ and $Y = y _ { 1 }$ take the form $\dot { X } = a e ^ { Y - \bar { X } } , \dot { Y } = b e ^ { - Y } - \tilde { a } e ^ { Y - X }$ with coeficients bounded above and below, and a logarithmic growth lemma for such pairs (Theorem $\mathrm { C . 4 } )$ shows that $e ^ { 2 Y - X }$ stays between two positive constants and yields $X = 2 \log ( \alpha t ) + O ( 1 ) , Y = \log ( \alpha t ) + O ( 1 )$ the bounds on u and v transfer these rates to $x _ { 2 }$ and $y _ { 2 }$ . Finally, since $\dot { g } _ { i } = - ( N - 1 ) \dot { d } _ { i }$ , the individual parameters are recovered as $\begin{array} { r } { g _ { 1 } = \frac { N - 1 } { N } y _ { 1 } + O ( 1 ) , d _ { 1 } = - \frac { 1 } { N } y _ { 1 } + O ( 1 ) } \end{array}$ , and likewise for $g _ { 2 } , d _ { 2 }$ (Equation (129)). Inserting these into the node-logit tables of Section C for an arbitrary graph and valid trace bounds every admissible-token loss by $O ( e ^ { - ( x _ { 2 } - y _ { 1 } ) } + e ^ { - y _ { 1 } } ) = O ( 1 / ( \alpha t ) )$ , except at a root parent, whose logit $\begin{array} { r } { g _ { 1 } - d _ { 1 } + d _ { 2 } = \frac { N - 1 } { N } \log ( \alpha t ) + O ( 1 ) } \end{array}$ gives Equation (18). □

![](images/47dd9b1b7b06f01f48e5395deae4f2f3b4d27f11f5fedb09132fff0feb558bcf.jpg)

![](images/9ef7c2cea5a5a0d969f02052a97865ded41851fbc8d4a76ea2ef4aad461046c7.jpg)  
Figure 1: Left: efective parameters against log t for the reduced flow (lines) and full-matrix training (markers); fitted slopes 1.99, 2.00, 1.00, 1.00 against the predicted 2, 2, 1, 1. Right: admissible-token loss on the test set by step type, and its maximum $\mathcal { L } _ { \mathrm { g e n } }$ . Backtracking to the root decays as $t ^ { - ( N - 1 ) / N } = t ^ { - 0 . 9 3 7 5 }$ ; all other steps decay as $t ^ { - 1 }$

## 5 Experiment

We validate the predictions of Section 4: the efective second-layer parameters grow logarithmically at the predicted rates, the gap variables stay bounded, and the generalization loss decays at the predicted polynomial rates. The protocol, the test set, the invariant-subspace check, and the initialization ablation are detailed in Section C.2.

Setup. We use $N = 1 6 , K = 8$ , the model of Section 4 with trainable $N \times N$ matrices $A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 }$ on top of the fixed first layer, the initialization Equation (16) with $c _ { 0 } = 1 / N ^ { 2 } , c _ { 0 } ^ { \prime } = c _ { 0 } / 2$ , and float64 arithmetic. Full-matrix training runs gradient descent with step size $\eta = 0 . 0 5$ on a sampled version of Equation (13) $( B = 1 2 8$ random vertex permutations per step, $1 0 ^ { 7 }$ steps, $t = 5 \times 1 0 ^ { 5 } , \alpha = 1 )$ The reduced flow integrates the eight-scalar system of Theorem C.1, which is exactly the averaged gradient flow on ${ \mathcal F }$ , up to $t = 1 0 ^ { 9 }$ . Generalization is measured on 64,325 DFS states from 300 random directed graphs on 16 vertices (with cycles and self-loops): 39,715 forward steps, 20,388 backtracking steps to a non-root parent, and 4,222 backtracking steps to the root.

Growth rates. Figure 1 (left) compares the learned efective parameters with logarithmic fits on $t \in [ 1 0 ^ { 3 } , 1 0 ^ { 9 } ]$ . For the reduced flow, $a _ { 2 }$ and $b _ { 1 }$ grow with slopes 1.992 and 2.000, and $g _ { 1 } - d _ { 1 } , g _ { 2 } - d _ { 2 }$ with slope 0.997; full-matrix training gives 1.966, 1.996, 0.990, 0.990 on its shorter horizon. The individual parameters follow Equation (129): $g _ { 1 } , g _ { 2 }$ grow with slope 0.935 (predicted $( N - 1 ) / N =$ 0.9375) and $d _ { 1 } , d _ { 2 }$ with slope −0.062 (predicted $- 1 / N )$ , while $g _ { 1 } + ( N - 1 ) d _ { 1 }$ is conserved to $1 0 ^ { - 1 1 }$ As Theorem C.5 predicts, $0 \leq a _ { 2 } - b _ { 1 } \leq 5 . 1$ and $| g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } | \leq 0 . 3 4$ along the whole trajectory, and $b _ { 2 } , d _ { 2 }$ become non-positive at $t \approx 0 . 3$ . The forward signal thus grows at twice the logarithmic rate of the backtracking signal, which is what lets the model prefer an unexplored child whenever one exists while keeping a lower-priority backtracking candidate. With $B = 1 2 8$ sampled permutations per step, the full matrices leave the invariant subspace $\mathcal { F }$ of Theorem 4.1 during the initial phase, when the dynamics slide along the switching surfaces (relative distance up to 0.9 for $B _ { 1 } , B _ { 2 } )$ , but return to within $6 \times 1 0 ^ { - 4 }$ of it by $t = 5 \times 1 0 ^ { 5 }$ , where their efective scalars agree with the reduced flow to within 0.03; larger batches $\left( B \ge 8 1 9 2 \right)$ keep the distance below 0.03 throughout (Section C.2).

Generalization. Figure 1 (right) shows the admissible-token loss by step type. On $t \in [ 1 0 ^ { 6 } , 1 0 ^ { 9 } ]$ the fitted exponents are −1.000 for forward steps, −1.000 for backtracking to a non-root parent, and −0.9375 for backtracking to the root, exactly the $- ( N - 1 ) / N$ of Theorem 4.2; the mechanism of Equation (129) is visible directly, since the logit of a root parent equals $g _ { 1 } - d _ { 1 } + d _ { 2 }$ to within 0.005. Repeating the reduced flow for other N (with $K = 3$ for $N = 8$ and $K = 4$ otherwise) gives fitted root-backtracking exponents
<table><tr><td>N</td><td>8</td><td>10</td><td>12</td><td>16</td><td>24</td><td>32</td></tr><tr><td>fitted</td><td>-0.8750</td><td>-0.9000</td><td>-0.9167</td><td>-0.9376</td><td>-0.9585</td><td>-0.9690</td></tr><tr><td> $- ( N - 1 ) / N$ </td><td>-0.8750</td><td>-0.9000</td><td>-0.9167</td><td>-0.9375</td><td>-0.9583</td><td>-0.9688</td></tr></table>

all within $3 \times 1 0 ^ { - 4 }$ of the prediction (Figure 4 in Section C.2 shows the local exponents). Over the whole observed range, however, the supremum ${ \mathcal { L } } _ { \mathrm { g e n } }$ is attained by forward steps at a node that also has an edge back to its stack parent: there the parent collects $g _ { 1 } - d _ { 1 }$ from the backtracking head and $a _ { 2 } - b _ { 1 }$ from the forward head, so the loss is $\approx e ^ { - ( b _ { 1 } - ( g _ { 1 } - d _ { 1 } ) ) } = \Theta ( e ^ { u } / t )$ , a $t ^ { - 1 }$ rate with the large constant $e ^ { u } \approx e ^ { 5 } ;$ the bound Equation (18) holds, but the $t ^ { - ( N - 1 ) / N }$ term overtakes this one only far beyond any practical horizon. The same states are the last to be classified correctly, at $t \approx 2 \times 1 0 ^ { 4 }$ , whereas all other test states are from $t \approx 4 0$ on (Section C.2).

## 6 Conclusion

We gave a theoretical account of how transformers implement, scale, and learn depth-first search on directed graphs. We established a sharp depth threshold for eficient DFS: any one-layer transformer requires $\Omega ( N ^ { 2 } )$ representation size (heads × width × precision) on dense graphs, while a two-layer construction sufices with O(N log N) representation size, where the first layer exposes predecessor information and the second combines competing forward and backtracking heads. Beyond this threshold, each additional layer doubles the radius of the bounded-reachability test available before every move, so the look-ahead horizon grows exponentially with depth. Finally, we proved that the two-layer mechanism emerges from gradient flow on a permutation-symmetric curriculum, with logarithmic growth rates and a polynomial generalization rate that our experiments reproduce quantitatively. Together, these results show that eficient DFS requires a genuine two-layer mechanism, that depth translates into exponentially larger look-ahead, and that such reasoning behavior can arise from gradient-based training on discrete chain-of-thought traces.

Limitations and future work. Our learnability analysis relies on a permutation-symmetric training curriculum and a fixed first layer; extending the gradient-flow analysis to end-to-end training over broader graph distributions is a natural next step. Our look-ahead construction is exact and has a fixed horizon, whereas trained large models appear to perform approximate, heuristic-driven search; closing this gap likely requires combining our discrete-CoT analysis with tools from approximate reasoning and value estimation. Finally, while DFS captures branching, memory, and backtracking, real reasoning tasks involve richer structures such as soft constraints, probabilistic transitions, and hierarchical subgoals. We hope the framework developed here—identifying depth thresholds, quantifying the gain from each additional layer, and tracking gradient-flow dynamics on structured curricula—can serve as a template for analyzing these more complex reasoning algorithms.

## 7 Acknowledgement

This work was supported by the U.S. Army Research Laboratory and the U.S. Army Research Ofice under Grant W911NF2010219, Ofice of Naval Research, and NSF. This work used Jetstream2 at Indiana University through allocation CIS251312 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## References

Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. In International Conference on Machine Learning, 2024.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. Advances in Neural Information Processing Systems, 36: 1560–1588, 2023.

Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Elias Bareinboim. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. arXiv preprint arXiv:2402.11917, 2024.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Unveiling induction heads: Provable training dynamics and feature learning in transformers. In Advances in Neural Information Processing Systems, 2024.

Benjamin L. Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, 2022.

Ezra Edelman, Benjamin L. Edelman, Surbhi Goel, Eran Malach, and Nikolaos Tsilivis. The evolution of statistical induction heads: In-context learning Markov chains. In Advances in Neural Information Processing Systems, 2024.

Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. In International Conference on Learning Representations, 2024.

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: A theoretical perspective. In Advances in Neural Information Processing Systems, 2023.

Arvid Frydenlund. The mystery of the pathological path-star task for language models. arXiv preprint arXiv:2410.13779, 2024.

Daya Guo et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics, 8:156–171, 2020.

Yixiao Huang, Hanlin Zhu, Tianyu Guo, Jiantao Jiao, Somayeh Sojoudi, Michael I. Jordan, Stuart Russell, and Song Mei. Generalization or hallucination? Understanding out-of-context reasoning in transformers. In Advances in Neural Information Processing Systems, 2025.

Yixiao Huang, Hanlin Zhu, Zixuan Wang, Jiantao Jiao, Stuart Russell, Somayeh Sojoudi, and Song Mei. Transformers provably learn to internalize chain-of-thought. arXiv preprint arXiv:2605.28600, 2026.

Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Kaya Stechly, Mudit Verma, Siddhant Bhambri, Lucas Saldyt, and Anil Murthy. LLMs can’t plan, but can help planning in LLM-Modulo frameworks. arXiv preprint arXiv:2402.01817, 2024.

Eyal Kushilevitz. Communication complexity. In Advances in Computers, volume 44, pages 331–360. Elsevier, 1997.

Lucas Lehnert, Sainbayar Sukhbaatar, DiJia Su, Qinqing Zheng, Paul McVay, Michael Rabbat, and Yuandong Tian. Beyond A<sup>∗</sup>: Better planning with transformers via search dynamics bootstrapping. arXiv preprint arXiv:2402.14083, 2024.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In International Conference on Learning Representations, 2024.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. In International Conference on Learning Representations, 2023.

Xutao Ma, Yixiao Huang, Hanlin Zhu, and Somayeh Sojoudi. Breaking the reversal curse in autoregressive language models via identity bridge. In International Conference on Machine Learning, 2026.

William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers. In Advances in Neural Information Processing Systems, 2023a.

William Merrill and Ashish Sabharwal. The parallelism tradeof: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531–545, 2023b.

William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. International Conference on Learning Representations 2024, 2024.

William Merrill and Ashish Sabharwal. Exact expressive power of transformers with padding. arXiv preprint arXiv:2505.18948, 2025.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations, 2023.

Eshaan Nichani, Alex Damian, and Jason D. Lee. How transformers learn causal structure with gradient descent. In International Conference on Machine Learning, 2024.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. Transformer Circuits Thread, 2022. arXiv:2209.11895.

Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. In Advances in Neural Information Processing Systems, 2024a.

Clayton Sanford, Daniel Hsu, and Matus Telgarsky. One-layer transformers fail to solve the induction heads task. arXiv preprint arXiv:2408.14332, 2024b.

Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. arXiv preprint arXiv:2402.09268, 2024c.

Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In International Conference on Learning Representations, 2023.

Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, and He He. Testing the general deductive reasoning capacity of large language models using OOD examples. In Advances in Neural Information Processing Systems, 2023.

Abulhair Saparov, Srushti Pawar, Shreyas Pimpalgaonkar, Nitish Joshi, Richard Yuanzhe Pang, Vishakh Padmakumar, Seyed Mehran Kazemi, Najoung Kim, and He He. Transformers struggle to learn to search. arXiv preprint arXiv:2412.04703, 2024.

Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. GPT-4 doesn’t know it’s wrong: An analysis of iterative prompting for reasoning problems. arXiv preprint arXiv:2310.12397, 2023.

Lena Strobl. Average-hard attention transformers are constant-depth uniform threshold circuits. arXiv preprint arXiv:2308.03212, 2023.

Yuandong Tian, Yiping Wang, Beidi Chen, and Simon S. Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer. In Advances in Neural Information Processing Systems, 2023.

Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models—a critical investigation. In Advances in Neural Information Processing Systems, 2023.

Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. Can language models solve graph problems in natural language? In Advances in Neural Information Processing Systems, 2023.

Zixuan Wang, Stanley Wei, Daniel Hsu, and Jason D. Lee. Transformers provably learn sparse token selection while fully-connected nets cannot. arXiv preprint arXiv:2406.06893, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022.

Noam Wies, Yoav Levine, and Amnon Shashua. Sub-task decomposition enables learning in sequence to sequence tasks. In International Conference on Learning Representations, 2023.

Tong Yang, Yu Huang, Yingbin Liang, and Yuejie Chi. Multi-head transformers provably learn symbolic multi-step reasoning via gradient descent. arXiv preprint arXiv:2508.08222, 2025. NeurIPS 2025.

Gilad Yehudai, Clayton Sanford, Maya Bechler-Speicher, Orr Fischer, Ran Gilad-Bachrach, and Amir Globerson. Depth-width tradeofs in algorithmic reasoning of graph tasks with transformers. arXiv preprint arXiv:2503.01805, 2025.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In International Conference on Learning Representations, 2020.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Emergence of superposition: Unveiling the training dynamics of chain of continuous thought. arXiv preprint arXiv:2509.23365, 2025a.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought. arXiv preprint arXiv:2505.12514, 2025b.

## A Related Work

Expressive power of transformers and chain of thought. A line of work characterizes what transformers can and cannot compute as a function of depth, precision, and CoT length. Hahn [2020] gives early limitations of self-attention on hierarchical and periodic structure, and Yun et al. [2020] shows universal approximation in the seq-to-seq sense. Tighter circuit-complexity bounds place log-precision transformers inside uniform ${ \mathsf { T C } } ^ { 0 }$ [Merrill and Sabharwal, 2023b,a, Strobl, 2023], and Merrill and Sabharwal [2025] extend these bounds with padding. On the positive side, Liu et al. [2023] show that low-depth transformers can simulate finite-state automata via Krohn–Rhodes shortcuts, and Edelman et al. [2022] characterize sparse-variable creation under bounded norm. CoT extends these limits: Feng et al. [2023] prove that constant-size transformers with CoT can solve dynamic programming and arithmetic that constant-depth transformers cannot, Li et al. [2024] show T CoT steps sufice to express any size-T circuit at constant precision, and Merrill and Sabharwal [2024] give a tight characterization of decoder transformers’ expressive power as the CoT length scales. On the learning side, Huang et al. [2026] analyze implicit CoT, where intermediate steps are removed during training so that the reasoning is absorbed into hidden states: a $\log _ { 2 }$ k-layer transformer trained under a curriculum that removes these steps in geometric chunks learns k-parity on n bits with poly(n) samples and $\log _ { 2 }$ k training stages, matching the sample eficiency of explicit CoT without generating thinking tokens at inference. Our results refine this landscape along the orthogonal axis of search-horizon-per-depth: we show DFS look-ahead grows exponentially in depth at near-linear width.

Graph reasoning with transformers. The graph-reasoning testbed has emerged as a natural arena for both empirical and theoretical analysis. Empirically, NLGraph [Wang et al., 2023], GraphQA [Fatemi et al., 2024], and the path-star analyses of Bachmann and Nagarajan [2024], Frydenlund [2024] expose systematic failures of LLMs on connectivity, shortest-path, and traversal tasks. Saparov et al. [2024] find that small transformers trained on DAG path-finding learn to maintain a per-vertex reachable set whose support grows exponentially with depth. Saparov and He [2023], Saparov et al. [2023] document analogous greedy-reasoning failures in deductive-proof settings. Theoretically, Sanford et al. [2024a] build a representational hierarchy of nine graph algorithms and show log-depth is necessary and suficient for connectivity; Sanford et al. [2024c] relate depth to MPC rounds; and Yehudai et al. [2025] establish depth–width tradeofs for graph primitives via 2-cycle communication-complexity arguments. Our work difers by isolating the algorithmic primitive (DFS, with explicit stack and backtracking) and proving a tight one-layer lower bound, a two-layer matching upper bound, and a depth–look-ahead hierarchy.

Provable training dynamics. A growing body of work analyzes gradient-based training of transformers on structured tasks. Tian et al. [2023] dissect token composition in one-layer transformers; Nichani et al. [2024] prove a two-layer transformer learns Markov-chain causal structure under gradient descent; Chen et al. [2024] and Olsson et al. [2022] provide training and circuit-level accounts of induction heads; Wang et al. [2024] prove transformers (but not MLPs) provably learn sparse token selection; and Edelman et al. [2024] analyze in-context Markov-chain learning. Another line explains what one-layer transformers infer beyond their training facts through the implicit bias of gradient descent: Huang et al. [2025] attribute out-of-context reasoning, which underlies both generalization and hallucination after fine-tuning, to a bias toward small nuclear norm of the combined output–value matrix, and Ma et al. [2026] prove that the reversal curse occurs on forward-only data but is broken when identity-bridge statements of the form $^ { 6 6 } A  A ^ { 5 }$ are added to the training set. Closest to our setting, Yang et al. [2025] prove that a one-layer multi-head transformer learns path-finding on trees via gradient descent with multi-head specialization. Trees, however, are search-trivial: there is at most one path between any pair of vertices, so neither branching nor backtracking arises. Our paper generalizes this line to arbitrary directed graphs by showing that two-layer transformers provably learn to execute DFS—the canonical algorithm that handles branching and revisitation—under gradient flow and generalize to unseen graphs. Zhu et al. [2025a] study a complementary regime: they prove that two-layer transformers learn directed-graph reachability with continuous CoT through the emergence of superposed reasoning traces. Whereas continuous CoT permits parallel exploration via embedding superposition, our discrete-CoT setting forces the model to produce a sequential search trace, and our analysis tracks how the canonical sequential-search algorithm (DFS) emerges in the parameters. Together, these two results paint a picture of how the same graph-search problem is solved diferently in the continuous and discrete CoT regimes.

Search and planning in language models. Beyond the symbolic setting, several empirical works probe whether LLMs perform search internally. Lehnert et al. [2024] train Searchformer to imitate $\mathrm { A } ^ { * }$ search and improve upon it via search-dynamics bootstrapping; Valmeekam et al. [2023], Kambhampati et al. [2024] document failure modes of LLMs as planners; and Stechly et al. [2023] analyze iterative-refinement limitations. Saparov and He [2023] show that LLMs are “greedy reasoners” that struggle when proofs branch. Mechanistic-interpretability work [Olsson et al., 2022, Nanda et al., 2023, Brinkmann et al., 2024] identifies internal circuits underlying simple reasoning and traversal behaviors, including a backward-chaining circuit on tree path-finding that motivated Yang et al. [2025]. Our theoretical results provide a complementary explanation for why transformers exhibit depth-dependent search horizons in these empirical studies: the k-layer DFS construction we exhibit has look-ahead horizon $1 + 2 ^ { k - 3 }$ , a depth-dependent horizon of the kind that empirical work observes, and our learnability result shows this construction is reachable by gradient-based training on appropriately curated CoT data.

## B Omitted proof in Section 3

## B.1 Proof of Theorem 3.3

Proof. We reduce from the one-way communication problem INDEX [Kushilevitz, 1997]. Alice receives

$$
f = ( f _ { 1 } , \dots , f _ { M } ) \in \{ 0 , 1 \} ^ { M } ,\tag{19}
$$

and Bob receives an index $i ^ { \star } \in [ M ]$ . Alice sends one message to Bob, who must output $f _ { i ^ { \star } }$ . Every deterministic one-way protocol for INDEX requires $\Omega ( M )$ bits.

Assume a one-layer transformer is correct on every valid instance of the unique outgoing-child task whose graph has vertex set [N] and exactly M distinct edges, where $N \geq 3 \lceil \sqrt { M } \rceil + 1$ . We construct a protocol for INDEX.

A row–column encoding with distinct edges. Set $m = \lceil \sqrt { M } \rceil$ and assign record $i \in [ M ]$ to the grid cell

$$
r ( i ) = 1 + \left\lfloor { \frac { i - 1 } { m } } \right\rfloor ,
$$

$$
c ( i ) = 1 + ( ( i - 1 ) \bmod m ) .\tag{20}
$$

The pairs $( r ( i ) , c ( i ) )$ are distinct and belong to $[ m ] \times [ m ]$ . Choose 3m + 1 pairwise distinct vertices in [N], named

$$
q , \quad a _ { 1 } , \ldots , a _ { m } , \quad b _ { 1 , 0 } , b _ { 1 , 1 } , \ldots , b _ { m , 0 } , b _ { m , 1 } .
$$

These names denote a fixed public labeling of the vertices. All remaining vertices, if any, are isolated. For each record $i \in [ M ]$ , define

$$
t _ { i } = b _ { c ( i ) , f _ { i } } ,
$$

$$
s _ { i } = { \left\{ \begin{array} { l l } { q , } & { i = i ^ { \star } , } \\ { a _ { r ( i ) } , } & { i \neq i ^ { \star } . } \end{array} \right. }\tag{21}
$$

Let

$$
\mathcal { E } = \{ ( s _ { i } \to t _ { i } ) : i \in [ M ] \} ,
$$

and list these edges in the public order $1 , \ldots , M$

We verify that $| { \mathcal { E } } | = M$ . The selected record is the only edge with source $q ,$ so it cannot coincide with any other edge. For two distinct unselected records $i , j , \mathrm { i f } \ r ( i ) \neq r ( j )$ , their sources are diferent. If $r ( i ) = r ( j )$ , then $c ( i ) \neq c ( j )$ , and their targets belong to the disjoint pairs $\big \{ b _ { c ( i ) , 0 } , b _ { c ( i ) , 1 } \big \}$ and $\left\{ b _ { c ( j ) , 0 } , b _ { c ( j ) , 1 } \right\}$ . Thus their targets are diferent. Every edge is therefore distinct. Moreover, the source vertices and target vertices are disjoint, so the construction has no self-loops.

The final query node is the fixed vertex $q .$ It has exactly one outgoing edge, whose target is

$$
t _ { i ^ { \star } } = b _ { c ( i ^ { \star } ) , f _ { i ^ { \star } } } .\tag{22}
$$

Since Bob knows $i ^ { \star }$ and the public vertex labeling, this output determines $f _ { i ^ { \star } }$

One-way simulation of the final position. The prompt has total length

$$
T = 3 M + 2 .\tag{23}
$$

Its final token is always $q ,$ at the same position, so its input embedding $\mathbfit { \mathbf { x } } _ { T }$ , including its positional encoding, is known to both parties and does not depend on $i ^ { \star }$ . Partition the positions into

$$
A = \{ 3 , 6 , 9 , \ldots , 3 M \} , \qquad B = [ T ] \setminus A .\tag{24}
$$

Alice controls the target-token positions A: each such token depends only on her input bit and its public column index. Bob knows all positions in $B _ { ; }$ , comprising the source tokens, special tokens, and final query. In particular, the input embedding at every position in A is independent of Bob’s index.

Fix one attention head $h ,$ and write

$$
A _ { h } = ( \boldsymbol { W } _ { K } ^ { ( 1 , h ) } ) ^ { \top } \boldsymbol { W } _ { Q } ^ { ( 1 , h ) } , \qquad \boldsymbol { w } _ { j } ^ { ( h ) } = \phi ( \boldsymbol { x } _ { j } ^ { \top } A _ { h } \boldsymbol { x } _ { T } ) .\tag{25}
$$

First consider unnormalized attention. The final column of head h is

$$
\operatorname { H e a d } _ { h } ^ { ( 1 ) } ( \mathbf { \boldsymbol { X } } ) _ { : , T } = \sum _ { j = 1 } ^ { T } w _ { j } ^ { ( h ) } \mathbf { \boldsymbol { W } } _ { V } ^ { ( 1 , h ) } \mathbf { \boldsymbol { x } } _ { j } .\tag{26}
$$

Alice computes

$$
S _ { A } ^ { ( h ) } = \sum _ { j \in A } w _ { j } ^ { ( h ) } W _ { V } ^ { ( 1 , h ) } { \pmb x } _ { j } \in \mathbb { R } ^ { d }\tag{27}
$$

and sends it to Bob. By the fixed-point convention in the definition of p-bit precision, $S _ { A } ^ { ( h ) }$ is a coordinate-wise sum of at most T numbers that are each p-bit fixed-point numbers, so it is exactly representable with $p + \lceil \log _ { 2 } T \rceil = O ( p )$ bits per coordinate, and the same holds for the partial sum over B below; no additional rounding is introduced by the protocol. The vector lives in the value space of the head, whose dimension is at most $d .$ Bob computes

$$
S _ { B } ^ { ( h ) } = \sum _ { j \in B } w _ { j } ^ { ( h ) } W _ { V } ^ { ( 1 , h ) } { \pmb x } _ { j }\tag{28}
$$

and reconstructs the head output as $S _ { A } ^ { ( h ) } + S _ { B } ^ { ( h ) }$ . The message has size $O ( d p )$ bits for this head. For normalized attention, the final column is

$$
\operatorname { H e a d } _ { h } ^ { ( 1 ) } ( { \mathbf { X } } ) _ { : , T } = { \frac { S _ { A } ^ { ( h ) } + S _ { B } ^ { ( h ) } } { Z _ { A } ^ { ( h ) } + Z _ { B } ^ { ( h ) } } } ,\tag{29}
$$

where

$$
Z _ { A } ^ { ( h ) } = \sum _ { j \in A } w _ { j } ^ { ( h ) } , \qquad Z _ { B } ^ { ( h ) } = \sum _ { j \in B } w _ { j } ^ { ( h ) } .\tag{30}
$$

Alice sends $S _ { A } ^ { ( h ) }$ and $Z _ { A } ^ { ( h ) }$ , each exactly representable with $O ( p )$ bits per coordinate by the same fixedpoint argument, and Bob computes the corresponding quantities for $B .$ . The additional normalizing scalar leaves the communication at $O ( d p )$ bits per head.

Repeating for all H heads, Alice sends $O ( H d p )$ bits. Bob reconstructs every final-column head output, applies the output matrix $\pmb { W } _ { O } ^ { ( 1 ) }$ and the residual update

$$
{ \pmb X } _ { : , T } ^ { ( 2 ) } = { \pmb x } _ { T } + \mathrm { R e L U } \Big ( \mathrm { A t t e n } ^ { ( 1 ) } ( { \pmb X } ) _ { : , T } \Big ) ,\tag{31}
$$

and computes the final logits $W _ { R } ^ { \top } X _ { : , T } ^ { ( 2 ) }$ . By correctness of the transformer, the predicted token is $b _ { c ( i ^ { \star } ) , f _ { i ^ { \star } } }$ , from which Bob recovers $f _ { i ^ { \star } }$ . Thus the transformer induces a one-way protocol for INDEX using $O ( H d p )$ bits. The $\Omega ( M )$ communication lower bound implies

$$
H d p = \Omega ( M ) .\tag{32}
$$

## B.2 Proof of Theorem 3.4

Proof. We have $3 \lceil \sqrt { M _ { N } } \rceil + 1 = 3 m _ { N } + 1 \leq N$ , so Theorem 3.3 applies. Since $M _ { N } = \Theta ( N ^ { 2 } )$ , it gives $H d p = \Omega ( N ^ { 2 } )$

For the DFS conclusion, use the same graph family from Section B.1, take $q$ as the root, and choose the fixed row vertex $a _ { 1 }$ as the target. Insert the public target token $a _ { 1 }$ and the public start-reasoning token $< r >$ immediately before the final root token $q .$ The initial DFS state is nonterminated and has visited set $\{ { q } \}$ . Its unique child is distinct from $q ,$ so every valid first DFS move must output that child. The additional task tokens are public and the final query remains fixed, so the same communication reduction applies. The target is unreachable in this family, which is permitted by the DFS task. □

## B.3 Token embeddings of the two-layer construction

We use the following token embeddings, excluding a fixed positional-embedding component:

$$
\mathrm { E d g e ~ s o u r c e } ~ s _ { i } : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { s _ { i } } \end{array} \right) , \qquad \mathrm { E d g e ~ t a r g e t } ~ t _ { i } : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { t _ { i } } \end{array} \right) , \qquad < { \sf s } > , < { \sf e } > , < { \sf r } > : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) ,\tag{33}
$$

$$
\mathrm { T a r g e t } ~ n _ { \mathrm { T a r } } : \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { n _ { \mathrm { T a r } } } \end{array} \right) , \qquad \mathrm { R o o t } ~ n _ { \mathrm { R o o t } } : \left( \begin{array} { l } { n _ { \mathrm { R o o t } } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) , \qquad \mathrm { C o T ~ s t e p } ~ n _ { t } : \left( \begin{array} { l } { n _ { t } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \end{array} \right) .\tag{34}
$$

Here $s _ { i } , t _ { i } , n _ { \mathrm { T a r } } .$ , and $n _ { t }$ denote the corresponding N-dimensional one-hot node embeddings. After the first layer of Theorem B.1, up to the scalar factors in $[ 1 - 2 \delta , 1 ]$ on the copied one-hot vectors, the hidden states in $X ^ { ( 2 ) }$ are those displayed in Equation (7).

On the embedding dimension. The token-embedding dimension is $O ( N )$ , which is natural for exact DFS because the algorithm must maintain and query an arbitrary visited set $S \subseteq [ N ]$ There are $2 ^ { N }$ possible visited sets, so any representation that preserves enough information for exact membership queries must distinguish $2 ^ { N }$ states. If the visited-set representation has dimension m and uses $p$ bits of precision per coordinate, then it can encode at most $2 ^ { O ( m p ) }$ distinguishable states, which implies $m p = \Omega ( N )$ With the $p = O ( \log N )$ precision used in our softmax construction, this yields $m = \Omega ( N / \log N )$ . Thus, the choice $m = O ( N )$ is within a logarithmic factor of the information-theoretic minimum for exact visited-set representation, while keeping the construction simple through one-hot node embeddings.

## B.4 Formal construction of the first layer

Lemma B.1 (Predecessor-copy layer). Let N be the number of graph nodes. Let $T _ { \mathrm { c o t } }$ be an upper bound on the length of the CoT trace and let $L _ { \mathrm { m a x } } = 3 M + T _ { \mathrm { c o t } } + 4$ , which bounds the total length of the prompt followed by the CoT trace (one $< s >$ token, 3M edge tokens, the three task tokens $n T a r ,$ $< r > , n _ { R o o t }$ , and at most $T _ { \mathrm { c o t } }$ CoT tokens); assume $M \leq N ^ { 2 }$ and $T _ { \mathrm { c o t } } \leq \mathrm { p o l y } ( N )$ . For each position $i \in [ L _ { \operatorname* { m a x } } ]$ , define

$$
p _ { i } = \left( \begin{array} { l } { \cos ( i \alpha ) } \\ { \sin ( i \alpha ) } \end{array} \right) , \qquad \alpha : = \frac { \pi } { 4 L _ { \mathrm { m a x } } } .\tag{35}
$$

Each token representation is

$$
\pmb { x } _ { i } = ( a _ { i } , b _ { i } , c _ { i } , d _ { i } , \rho _ { i } , p _ { i } ) \in \mathbb { R } ^ { 4 N + 3 } , \qquad \rho _ { i } = 1 ,\tag{36}
$$

where $a _ { i } , b _ { i } , c _ { i } , d _ { i } \in \mathbb { R } ^ { N }$ . The scalar coordinate ρ is a constant technical coordinate. Consider the $\rho _ { i }$ first layer

$$
{ \pmb X } ^ { ( 2 ) } = { \pmb X } ^ { ( 1 ) } + R e L U \Bigl ( A t t e n ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) \Bigr ) ,\tag{37}
$$

$$
\begin{array} { r } { A t t e n ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) = { \pmb W } _ { O } ^ { ( 1 ) } \left[ \mathrm { H e a d } _ { 1 } ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) \right] , } \end{array}\tag{38}
$$

$$
\mathrm { H e a d } _ { h } ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) = { \pmb W } _ { V } ^ { ( 1 , h ) } { \pmb X } ^ { ( 1 ) } \phi ^ { ( 1 ) } \Bigl ( ( { \pmb X } ^ { ( 1 ) } ) ^ { \top } ( { \pmb W } _ { K } ^ { ( 1 , h ) } ) ^ { \top } { \pmb W } _ { Q } ^ { ( 1 , h ) } { \pmb X } ^ { ( 1 ) } \Bigr ) ,\tag{39}
$$

where $\phi ^ { ( 1 ) } = \mathrm { s o f t m a x } _ { \mathrm { c o l } }$ is column-wise softmax. Then for any $\delta \in ( 0 , 1 / 8 )$ , there exist matrices

$$
\{ { \cal W } _ { Q } ^ { ( 1 , h ) } , { \cal W } _ { K } ^ { ( 1 , h ) } , { \cal W } _ { V } ^ { ( 1 , h ) } \} _ { h = 1 } ^ { 3 } , \qquad { \cal W } _ { O } ^ { ( 1 ) } ,
$$

with embedding dimension $d = 4 N + 3 = O ( N )$ , such that for every position $i \geq 2 .$

1. if $a _ { i - 1 } = u$ is a one-hot vector, then the second block added by the first layer is supported exactly on $\operatorname { s u p p } ( u )$ and equals λu for some $\lambda \in \left[ 1 - 2 \delta , 1 \right]$ ;

2. if $a _ { i - 1 } = 0 _ { N }$ , then the second block added by the first layer is exactly $0 _ { N }$ ;

3. if $d _ { i - 1 } = v$ is a one-hot vector, then the third block added by the first layer is supported exactly on supp(v) and equals λv for some $\lambda \in \left[ 1 - 2 \delta , 1 \right]$ ;

4. if $d _ { i - 1 } = 0 _ { N }$ , then the third block added by the first layer is exactly $0 _ { N }$

Equivalently, for the tokens used in the DFS prompt,

$$
\mathrm { E d g e } \ s _ { i } = \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { s _ { i } } \\ { 1 } \\ { p } \end{array} \right) , \mathrm { E d g e } \ t _ { i } = \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { t _ { i } } \\ { 1 } \\ { p } \end{array} \right) , < s > , < e > , < r > = \left( \begin{array} { l } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) ,\tag{40}
$$

$$
\mathrm { T a r g e t } = \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { n _ { T a r } } \\ { 1 } \\ { p } \end{array} \right) , \mathrm { R o o t } = \left( \begin{array} { c } { n _ { R o o t } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) , \mathrm { C o T } \ t = \left( \begin{array} { c } { n _ { t } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) ,\tag{41}
$$

the first layer gives

$$
\begin{array} { r } { \mathrm { E d g e } ~ s _ { i } \longmapsto \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { s _ { i } } \\ { 1 } \\ { p } \end{array} \right) , \quad \mathrm { R o o t } \longmapsto \left( \begin{array} { c } { n _ { R o o t } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) , \quad \mathrm { C o T } ~ t \longmapsto \left( \begin{array} { c } { n _ { t } } \\ { ( 1 \pm O ( \delta ) ) n _ { t - 1 } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) , } \end{array}\tag{42}
$$

$$
\mathrm { E d g e } ~ t _ { i } \longmapsto \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { ( 1 \pm O ( \delta ) ) s _ { i } } \\ { t _ { i } } \\ { 1 } \\ { p } \end{array} \right) , \quad \mathrm { T a r g e t } \longmapsto \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { 0 _ { N } } \\ { n _ { T a r } } \\ { 1 } \\ { p } \end{array} \right) , \quad < e > \longmapsto \left( \begin{array} { c } { 0 _ { N } } \\ { 0 _ { N } } \\ { ( 1 \pm O ( \delta ) ) t _ { i } } \\ { 0 _ { N } } \\ { 1 } \\ { p } \end{array} \right) ,\tag{43}
$$

$$
< r > \longmapsto \left( \begin{array} { c } { { 0 _ { N } } } \\ { { 0 _ { N } } } \\ { { ( 1 \pm O ( \delta ) ) n _ { T a r } } } \\ { { 0 _ { N } } } \\ { { 1 } } \\ { { p } } \end{array} \right) , < s > \longmapsto \left( \begin{array} { c } { { 0 _ { N } } } \\ { { 0 _ { N } } } \\ { { 0 _ { N } } } \\ { { 0 _ { N } } } \\ { { 1 } } \\ { { p } } \end{array} \right) ,\tag{44}
$$

Here every factor $( 1 \pm O ( \delta ) )$ denotes a scalar in $[ 1 - 2 \delta , 1 ]$ , and the supports of the copied one-hot vectors are exact after the ReLU filter. The claim $f o r < s >$ at position 1, which has no predecessor, is included: that position attends to itself and all of its node blocks are zero. If $L _ { \mathrm { m a x } } = \mathrm { p o l y } ( N )$ and $\delta \ge 1 / \mathrm { p o l y } ( N )$ , then the construction uses ${ \cal O } ( \log N )$ -bit parameter magnitudes and precision.

Proof. We give an explicit construction.

Step 1: a softmax predecessor selector. Let

$$
R _ { - 1 } : = \left( \begin{array} { l l } { \cos \alpha } & { \sin \alpha } \\ { - \sin \alpha } & { \cos \alpha } \end{array} \right) .\tag{45}
$$

Then $R _ { - 1 } p _ { i } = p _ { i - 1 }$ for all $i \geq 2$ . For each head $h \in \{ 1 , 2 , 3 \}$ , choose ${ \pmb W } _ { \boldsymbol { Q } } ^ { ( 1 , h ) }$ and ${ W _ { K } ^ { ( 1 , h ) } }$ to ignore the four node blocks and the scalar coordinate, and to act on the positional block as

$$
\begin{array} { r } { W _ { Q } ^ { ( 1 , h ) } \pmb { x } _ { i } = \beta R _ { - 1 } p _ { i } , \qquad W _ { K } ^ { ( 1 , h ) } \pmb { x } _ { j } = p _ { j } . } \end{array}\tag{46}
$$

Thus the score from key position $j$ to query position i is

$$
S _ { j , i } = \beta \langle p _ { j } , p _ { i - 1 } \rangle = \beta \cos ( ( j - i + 1 ) \alpha ) .\tag{47}
$$

For fixed $i \geq 2$ , the unique maximizer is $j = i - 1$ . Moreover, for $j \neq i - 1$

$$
S _ { i - 1 , i } - S _ { j , i } = \beta \left( 1 - \cos ( ( j - i + 1 ) \alpha ) \right) \geq \beta ( 1 - \cos \alpha ) .\tag{48}
$$

Since $\alpha = \pi / ( 4 L _ { \mathrm { m a x } } )$ , there is a universal constant $c > 0$ such that

$$
1 - \cos \alpha \geq \frac { c } { L _ { \mathrm { m a x } } ^ { 2 } } .\tag{49}
$$

Choose

$$
\beta \ge C L _ { \operatorname* { m a x } } ^ { 2 } \log ( L _ { \operatorname* { m a x } } / \delta )\tag{50}
$$

with a suficiently large universal constant C. Then the column-wise softmax attention matrix P satisfies

$$
P _ { i - 1 , i } \geq 1 - \delta , \qquad \sum _ { j \neq i - 1 } P _ { j , i } \leq \delta .\tag{51}
$$

Step 2: value maps. All three heads use the same softmax predecessor matrix P. The first head extracts the first node block:

$$
\begin{array} { r } { W _ { V } ^ { ( 1 , 1 ) } \pmb { x } _ { j } = a _ { j } . } \end{array}\tag{52}
$$

The second head extracts the fourth node block:

$$
\begin{array} { r } { W _ { V } ^ { ( 1 , 2 ) } \pmb { x } _ { j } = d _ { j } . } \end{array}\tag{53}
$$

The third head uses the constant scalar coordinate to produce a threshold vector:

$$
\begin{array} { r } { { \cal W } _ { V } ^ { ( 1 , 3 ) } { \cal x } _ { j } = \rho _ { j } { \bf 1 } _ { N } = { \bf 1 } _ { N } . } \end{array}\tag{54}
$$

Because every column of P sums to one, the third head outputs exactly ${ \bf 1 } _ { N }$ at every position:

$$
\Big ( \mathrm { H e a d } _ { 3 } ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) \Big ) _ { i } = { \bf 1 } _ { N } .\tag{55}
$$

Step 3: output map and ReLU filter. Let the output matrix place a thresholded copy of the first head into the second node block and a thresholded copy of the second head into the third node block:

$$
\begin{array} { r } { W _ { O } ^ { ( 1 ) } \left[ \begin{array} { c } { u } \\ { v } \\ { r } \end{array} \right] = \left( \begin{array} { c } { 0 _ { N } } \\ { 2 u - r } \\ { 2 v - r } \\ { 0 _ { N } } \\ { 0 } \\ { 0 _ { 2 } } \end{array} \right) , \qquad u , v , r \in \mathbb { R } ^ { N } . } \end{array}\tag{56}
$$

Hence the pre-ReLU attention output at position i is

$$
\begin{array} { r } { z _ { i } : = \mathrm { A t t e n } ^ { ( 1 ) } ( { \pmb X } ^ { ( 1 ) } ) _ { i } = \left( \begin{array} { c } { 0 _ { N } } \\ { 2 \sum _ { j } P _ { j , i } a _ { j } - \mathbf { 1 } _ { N } } \\ { 2 \sum _ { j } P _ { j , i } d _ { j } - \mathbf { 1 } _ { N } } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right) . } \end{array}\tag{57}
$$

Now suppose $a _ { i - 1 } = e _ { k }$ is one-hot. By $\operatorname { E q }$ . equation 51, the k-th coordinate of $\sum _ { j } P _ { j , i } a _ { j }$ lies in $[ 1 - \delta , 1 ] \colon$ it is at least $P _ { i - 1 , i } \ge 1 - \delta$ , and it is at most one because the attention weights sum to one and every $a _ { j }$ has entries in $\{ 0 , 1 \}$ . Every coordinate outside k is at most $\delta ,$ , because the total non-predecessor mass is at most δ. Hence

$$
\mathrm { R e L U } \left( 2 \sum _ { j } P _ { j , i } a _ { j } - \mathbf { 1 } _ { N } \right) = \lambda e _ { k } , \qquad \lambda \in [ 1 - 2 \delta , 1 ] ,\tag{58}
$$

with exact support $\{ k \}$ , because $2 \delta - 1 < 0$ kills every other coordinate. If $a _ { i - 1 } = 0 _ { N }$ , then every coordinate of $\textstyle \sum _ { j } P _ { j , i } a _ { j }$ is at most $\delta < 1 / 2$ , and therefore

$$
\mathrm { R e L U } \left( 2 \sum _ { j } P _ { j , i } a _ { j } - \mathbf { 1 } _ { N } \right) = 0 _ { N } .\tag{59}
$$

The same argument applied to the fourth node block proves the corresponding claims for $d _ { i - 1 }$ and the third block.

Step $\it 4 \mathrm { : }$ applying to the prompt. For an edge target token Edge $t _ { i } ,$ the previous token is Edge $s _ { i } ,$ whose fourth node block is $s _ { i }$ and whose first node block is zero. Thus the ReLU-filtered residual addition places $( 1 \pm O ( \delta ) ) s _ { i }$ into the third block and places zero into the second block. For the source token, the previous token is $< \mathsf { e } > ,$ , whose four node blocks are all zero, so the ReLU-filtered residual addition is zero. For a CoT token $n _ { t } .$ , the previous token has first node block $n _ { t - 1 }$ and fourth node block zero, so the ReLU-filtered residual addition places $( 1 \pm O ( \delta ) ) n _ { t - 1 }$ into the second block and places zero into the third block. Finally, at position $i = 1$ the scores are $S _ { j , 1 } = \beta \cos ( j \alpha )$ for $j \geq 1$ , whose unique maximizer is $j = 1$ with gap $S _ { 1 , 1 } - S _ { j , 1 } \geq \beta ( \cos \alpha - \cos 2 \alpha ) \geq \beta ( 1 - \cos \alpha )$ hence the same concentration bound holds with $i - 1$ replaced by 1, and since $a _ { 1 } = d _ { 1 } = 0 _ { N }$ for ${ \tt } < { \tt } { \tt { s } } >$ both ReLU-filtered additions at position 1 are $0 _ { N }$

Finally, $d = 4 N + 3 = O ( N )$ . The nonzero parameters are 0, 1, 2, the rotation entries cos $\alpha .$ sin $\alpha _ { \mathrm { { \ell } } }$ and the scale $\beta .$ . If $L _ { \mathrm { m a x } } = \mathrm { p o l y } ( N )$ and $\delta \ge 1 / \mathrm { p o l y } ( N )$ , then $\beta = \mathrm { p o l y } ( N )$ and all parameters can be represented to inverse-polynomial accuracy with ${ \cal O } ( \log N )$ bits. The softmax concentration bound is stable under inverse-polynomial perturbations of the scores, so ${ \cal O } ( \log N )$ precision sufices.

## B.5 Proof of Theorem 3.6

Proof. The representation-size bound follows from Theorem B.1: $O ( 1 )$ heads, embedding dimension $4 N + 3 .$ , and ${ \cal O } ( \log N )$ precision, the second-layer weights being constants. The lemma applies with $L _ { \operatorname* { m a x } } = 3 N ^ { 2 } + 2 N + 2 = \mathrm { p o l y } ( N )$ to every graph on [N] and every valid DFS trace on it. Indeed $M \leq N ^ { 2 }$ , and a valid DFS trace $( n _ { 0 } , \ldots , n _ { t } )$ has $t \leq 2 N - 2$ : each forward move enters a previously unvisited vertex, so there are at most $N - 1$ forward moves, and each backtracking step pops a vertex pushed by an earlier forward move, so there are at most as many backtracking steps as forward moves.

Fix a valid nonterminated DFS trace and let $n _ { t }$ be the current node. By Theorem B.1, every edge-target token exposes its source in the third block and its destination in the fourth block, and every CoT token exposes the current node in the first block and the previous node in the second block, where each copied one-hot vector carries a scalar factor in $[ 1 - 2 \delta , 1 ]$ and all other coordinates of the copied block are exactly zero. The forward head assigns score $c _ { 1 } \lambda$ with $\lambda \in [ 1 - 2 \delta , 1 ]$ to each edge-target token whose source is $n _ { t }$ , and score exactly $c _ { 1 }$ to each CoT token, including the root; its value map adds the destination of an edge-target token and subtracts the node of a CoT token. Hence, before the ReLU, an unvisited child of $n _ { t }$ has forward score at least $c _ { 1 } ( 1 - 2 \delta )$ , a visited child has forward score at most $c _ { 1 } ( 1 - 1 ) = 0$ , and a non-child has forward score at most 0. The backtracking head attends to the CoT tokens that enter or leave $n _ { t }$ . Since $n _ { t }$ is the top of the current DFS stack, it has never been popped. Hence every step that left $n _ { t }$ was a forward move to a previously unvisited child, each such child was entered exactly once, and its excursion has been completed by the backtracking step that returned to $n _ { t }$ . In particular, no step went directly from $n _ { t }$ to its stack parent, which was visited before $n _ { t }$ and is therefore neither an unvisited child of $n _ { t }$ nor a backtracking target from $n _ { t }$ while $n _ { t }$ is on the stack. Consequently every completed excursion from $n _ { t }$ to a child and back contributes exactly one departure term and one return term, whose node coordinates cancel up to the leakage, leaving at most $2 \delta c _ { 2 }$ on that child, while the stack parent receives only the unique tree-edge arrival into $n _ { t }$ , which contributes at least $c _ { 2 } ( 1 - 2 \delta )$ and at most $c _ { 2 }$ on the stack parent and a negative amount on $n _ { t } ;$ at the root there is no tree-edge arrival. After the ReLU and the readout, every unvisited child has logit at least $c _ { 1 } ( 1 - 2 \delta )$ , the stack parent has logit in $[ c _ { 2 } ( 1 - 2 \delta ) , c _ { 2 } ]$ , and every other vertex has logit at most $2 \delta c _ { 2 }$ . By Equation (9), $c _ { 1 } ( 1 - 2 \delta ) > c _ { 2 }$ and $\delta < 1 / 4$ gives $c _ { 2 } ( 1 - 2 \delta ) > 2 \delta c _ { 2 }$ . Therefore, if $n _ { t }$ has an unvisited child, every maximizer is an unvisited child of $n _ { t } ;$ otherwise $n _ { t }$ is not the root, because the trace is nonterminated, and the unique maximizer is the stack parent. In both cases every maximizer of the next-token distribution is a valid DFS move. □

## B.6 Proof of the look-ahead construction and a conditional eficiency bound

Lemma B.2 (k-layer transformer look-ahead). For every integer $k \geq 3$ , there exists a k-layer transformer with unmasked attention that implements the bounded-reachability policy in Theorem 3.7 with horizon $L = 1 + 2 ^ { k - 3 }$ . Its token embeddings consist of node one-hot blocks together with the positional encoding $p _ { i }$ of Theorem B.1, a constant coordinate, a beginning-of-sequence indicator, and a period-three position indicator, as specified in the proof. The construction has $O ( 1 )$ heads per layer, width $O ( N )$ , and fixed-point precision $O _ { k } ( \log ( N + 1 ) )$ .

Proof. We specify the hidden state after each layer and the attention heads producing its next update. Write $g = n _ { \mathrm { T a r } }$ , and let $\boldsymbol { e } _ { w } \in \mathbb { R } ^ { N }$ be the one-hot vector of vertex $w .$ . An edge record is the target-token position in a triple $( u , v , < \mathsf { e } > )$ . All attention sums below range over the entire current input prefix, of length $T$

State notation and permitted updates. Let $h _ { i } ^ { [ \ell ] } = X _ { : , i } ^ { ( \ell + 1 ) }$ denote the state at position i after ℓ layers, so $h _ { i } ^ { [ 0 ] }$ is the input embedding. The architecture updates it by

$$
h _ { i } ^ { [ \ell + 1 ] } = h _ { i } ^ { [ \ell ] } + \mathrm { R e L U } \Big ( W _ { O } ^ { [ \ell + 1 ] } [ H _ { 1 , i } ; \ldots ; H _ { H , i } ] \Big ) .\tag{60}
$$

For a linear-attention head, a row $\left( q _ { i } , k _ { s } , v _ { s } \right)$ in the tables below means

$$
H _ { i } = \sum _ { s = 1 } ^ { T } \langle q _ { i } , k _ { s } \rangle v _ { s } .
$$

Each displayed query, key, and value is a linear function of the incoming state; smaller vectors are padded by zeros to a common head dimension. The output projection routes and linearly combines these head outputs in the specified coordinate blocks; the only post-attention nonlinearity is the single ReLU in Equation (60).

For exact logical copies, use a pair of physical blocks

$$
\widehat { Y } = ( Y ^ { + } , Y ^ { - } ) , \qquad Y = Y ^ { + } - Y ^ { - } .
$$

A query, key, value, or readout map can decode Y by a linear subtraction. For a nonnegative increment $\Delta ,$ write

$$
\widehat { Y } \oplus \Delta = ( Y ^ { + } + \Delta , Y ^ { - } ) .
$$

This is a residual update to the plus block, leaving the minus block unchanged. Also write

$$
\Pi ( z ) = \bigl ( \mathrm { R e L U } ( z ) , \mathrm { R e L U } ( z - { \bf 1 } ) \bigr ) .
$$

When written into two initially zero blocks, $\Pi ( z )$ is realized by two rows of the same output projection followed by the layer’s single ReLU. Neither notation introduces an extra layer.

Retain the input block

$$
\eta _ { i } = ( a _ { i } , d _ { i } , \gamma _ { i } , \rho _ { i } , \chi _ { i } , \zeta _ { i } , p _ { i } ) .
$$

Here $a _ { i }$ is the node one-hot at root and CoT positions and zero elsewhere; $d _ { i }$ is the node one-hot at edge endpoints and the task-target position and zero elsewhere; and $\gamma _ { i } = e _ { g }$ only at the task-target position. Furthermore, $\rho _ { i } = 1 , \ \zeta _ { i }$ is the BOS indicator, $\chi _ { i } = 1 \{ i \equiv 0 { \pmod { 3 } } \}$ , and $p _ { i }$ is the two-dimensional positional encoding from Theorem B.1. The features use $O ( N )$ coordinates and remain unchanged throughout. An edge record is characterized by $\chi _ { i } = 1$ and $\mathbf { 1 } ^ { \top } d _ { i } = 1$ . Indeed, the edge-target token of the i-th edge sits at position 3i; the task target sits at position $3 M + 2$ and has $\chi = 0 ;$ the token $< r >$ sits at position $3 M + 3$ and has $\chi = 1$ but $d = 0 ;$ ; and every root or CoT position, as well as every source and ${ \tt < e > }$ position, has $d = 0 \ \mathrm { o r } \ \chi = 0$

The complete physical state has the fixed coordinate layout

$$
h _ { i } ^ { [ \ell ] } = \big ( \eta _ { i } , \widehat { b } _ { i } , \widehat { U } _ { i } , \widehat { V } _ { i } , \widehat { G } _ { i } , \widehat { S } _ { i } , \widehat { C } _ { i } , D _ { i } ^ { \mathrm { f } } , D _ { i } ^ { \mathrm { b } } , Z _ { i } ^ { + } , Z _ { i } ^ { - } \big ) ,\tag{61}
$$

where the blocks after $\eta _ { i }$ are initially zero. The paired blocks store the predecessor, source identifier, destination identifier, goal, source reachability memory, and child reachability memory, respectively. The last four blocks hold the forward DFS score, backtracking score, and two final eligibility outputs. Each unpaired working block has N coordinates; each paired block has 2N coordinates.

Layer 1: endpoint and goal copies. After the first layer, the exact decoded values are

$$
( a _ { i } , b _ { i } , U _ { i } , V _ { i } , G _ { i } ) = \{ \begin{array} { l l } { ( 0 , 0 , e _ { u } , e _ { v } , e _ { g } ) , } & { i \mathrm { ~ r e c o r d s ~ } ( u  v ) , } \\ { ( e _ { n _ { \mathrm { R o o t } } } , 0 , 0 , e _ { g } ) , } & { i \mathrm { ~ i s ~ t h e ~ r o o t ~ p o s i t i o n } , } \\ { ( e _ { n _ { t } } , e _ { n _ { t - 1 } } , 0 , 0 , e _ { g } ) , } & { i \mathrm { ~ i s ~ C o T ~ s t e p ~ } t , } \\ { ( 0 , 0 , 0 , 0 , e _ { g } ) , } & { \mathrm { o t h e r w i s e } , } \end{array}\tag{62}
$$

where $n _ { 0 }$ is the root. Define the immutable block

$$
I _ { i } = ( \eta _ { i } , \widehat { b } _ { i } , \widehat { U } _ { i } , \widehat { V } _ { i } , \widehat { G } _ { i } ) .
$$

Duplicate the source and destination pairs into the two mutable memories. Thus the full state after layer 1 is

$$
\boxed { h _ { i } ^ { [ 1 ] } = ( I _ { i } , \widehat { U } _ { i } , \widehat { V } _ { i } , 0 , 0 , 0 , 0 ) . }\tag{63}
$$

In this display the second and third entries occupy the $\widehat { S }$ and $\widehat { C }$ coordinates of Equation (61); they are separate copies of the pairs retained in $I _ { i }$

Here is an explicit realization of this state. The first layer uses the following three softmax heads:

<table><tr><td>head</td><td> $q _ { i }$ </td><td> $k _ { s }$ </td><td>Us</td></tr><tr><td>predecessor</td><td> $\overline { { \beta { R _ { - 1 } } { p _ { i } } } }$ </td><td> $p _ { s }$ </td><td>(ds, as, ρs)</td></tr><tr><td>self</td><td>βpi</td><td>ps</td><td>(ds, as, Xs, ρs)</td></tr><tr><td>goal</td><td>βgρi</td><td>1+γs</td><td>(γs, ρs)</td></tr></table>

The predecessor selector is that of Theorem B.1; the self selector uses the identity rotation. Choose the parameters and rounding accuracy so that their outputs satisfy

$$
z _ { i } ^ { \mathrm { p r e } } \approx d _ { i - 1 } , \quad h _ { i } ^ { \mathrm { p r e } } \approx a _ { i - 1 } , \quad z _ { i } ^ { \mathrm { s e l f } } \approx d _ { i } , \quad h _ { i } ^ { \mathrm { s e l f } } \approx a _ { i } , \quad c _ { i } \approx \chi _ { i } , \quad z _ { i } ^ { \mathrm { g o a l } } \approx e _ { g } ,
$$

with efective error at most $\delta < 1 / 6 4$ on every used coordinate and scalar sum. The predecessor approximations are needed only for $i > 1$ . Each head’s constant-value output is one. Set $\bar { n } _ { i } = \mathbf { 1 } ^ { \top } z _ { i } ^ { \mathrm { s e l f } }$ and $t _ { i } = \mathbf { 1 } ^ { \top } h _ { i } ^ { \mathrm { s e l f } }$ . The output projection forms

$$
\begin{array} { c } { { \omega _ { i } ( z ) = 4 z + 4 ( c _ { i } + n _ { i } ) \mathbf { 1 } - 9 \mathbf { 1 } , } } \\ { { { } } } \\ { { r _ { i } = 4 h _ { i } ^ { \mathrm { p r e } } + 4 t _ { i } \mathbf { 1 } - 5 \mathbf { 1 } , } } \\ { { { } } } \\ { { g _ { i } ^ { \mathrm { c o p y } } = 4 z _ { i } ^ { \mathrm { g o a l } } - \mathbf { 1 } , } } \end{array}
$$

and writes

$$
\widehat { U } _ { i } = \Pi ( \omega _ { i } ( z _ { i } ^ { \mathrm { p r e } } ) ) , \qquad \widehat { V } _ { i } = \Pi ( \omega _ { i } ( z _ { i } ^ { \mathrm { s e l f } } ) ) , \qquad \widehat { b } _ { i } = \Pi ( r _ { i } ) , \qquad \widehat { G } _ { i } = \Pi ( g _ { i } ^ { \mathrm { c o p y } } ) .
$$

The same outputs for $\widehat { U } _ { i } , \widehat { V } _ { i }$ are routed to the two mutable memories. At a desired one-hot coordinate the argument of Π is close to $3 ;$ at any other coordinate it is at most $- 1 + 1 2 \delta < 0$ . The diference of the two physical outputs is therefore exactly one or zero, respectively. For endpoint copies, the three tests are the endpoint coordinate, the record-position feature, and the presence of a node in $d _ { i } .$ . This excludes source tokens, the task target, special tokens, and CoT tokens from the edge memories. All ofsets are linear maps of the constant-value output, and all pairs are produced in parallel by one ReLU. The finite-precision justification is given below.

Layer 2: one-step memories and the DFS scores. All subsequent attention is unnormalized linear attention. Use four heads, with $\boldsymbol { \tau } _ { s } = \mathbf { 1 } ^ { \top } a _ { s } \mathrm { : }$
<table><tr><td>head</td><td> $q _ { i }$ </td><td> $k _ { s }$ </td><td> $v _ { s }$ </td><td>output block</td></tr><tr><td>source neighbors</td><td> $U _ { i }$ </td><td> $U _ { s }$ </td><td> $\overline { { V _ { s } } }$ </td><td> $\overline { { S _ { i } ^ { + } } }$ </td></tr><tr><td>child neighbors</td><td> $V _ { i }$ </td><td> $U _ { s }$ </td><td> $V _ { s }$ </td><td> $\dot { C } _ { i } ^ { + }$ </td></tr><tr><td>DFS forward</td><td> $a _ { i }$ </td><td> $U _ { s } + \tau _ { s } \mathbf { 1 }$ </td><td> $2 ( V _ { s } - a _ { s } )$ </td><td> $\dot { D _ { i } ^ { \mathrm { f } } }$ </td></tr><tr><td>DFS backtrack</td><td> $a _ { i }$ </td><td> $a _ { s } + b _ { s }$ </td><td> $b _ { s } - a _ { s }$ </td><td> $D _ { i } ^ { \mathrm { { \dot { b } } } }$ </td></tr></table>

(64)

Every entry is a linear map of $h ^ { [ 1 ] }$ . Route each head to the indicated block and set all other rows of the output projection to zero. Consequently,

$$
\widehat { S } _ { i } ^ { ( 0 ) } = \widehat { U } _ { i } \oplus \sum _ { s } \langle U _ { i } , U _ { s } \rangle V _ { s } ,
$$

$$
\widehat C _ { i } ^ { ( 0 ) } = \widehat V _ { i } \oplus \sum _ { s } \langle V _ { i } , U _ { s } \rangle V _ { s } ,\tag{65}
$$

$$
D _ { i } ^ { \mathrm { f } } = \mathrm { R e L U } \left( 2 \sum _ { s } \langle a _ { i } , U _ { s } + \tau _ { s } { \bf 1 } \rangle ( V _ { s } - a _ { s } ) \right) ,\tag{66}
$$

$$
D _ { i } ^ { \mathrm { b } } = \mathrm { R e L U } \left( \sum _ { s } \langle a _ { i } , a _ { s } + b _ { s } \rangle ( b _ { s } - a _ { s } ) \right) .\tag{67}
$$

The memory increments are nonnegative, so the ReLU leaves them unchanged. The score blocks start at zero and receive the ReLU of their respective head outputs. The full hidden state is now

$$
\boxed { h _ { i } ^ { [ 2 ] } = ( I _ { i } , \widehat { S } _ { i } ^ { ( 0 ) } , \widehat { C } _ { i } ^ { ( 0 ) } , D _ { i } ^ { \mathrm { f } } , D _ { i } ^ { \mathrm { b } } , 0 , 0 ) . }\tag{68}
$$

For the record $\boldsymbol { e } = \left( u , v \right)$ , its decoded memories are explicitly

$$
S _ { e } ^ { ( 0 ) } = e _ { u } + \sum _ { ( u  w ) \in \mathcal { E } } e _ { w } ,\tag{69}
$$

$$
C _ { e } ^ { ( 0 ) } = e _ { v } + \sum _ { ( v  w ) \in \mathscr { E } } e _ { w } .\tag{70}
$$

The endpoint terms come from the residual connection, and the neighbor sums come from the first two attention heads. At nonrecord positions both memories are zero.

At the final position $i = T$ , let x be the current node. The last two heads are the ordinary DFS heads in Equation (8) with $c _ { 1 } = 2 , c _ { 2 } = 1$ , now written as explicit projections into fresh blocks. The forward head subtracts two times the occurrence count of each visited node from two times the adjacency vector of $x .$ In the backtracking head, completed excursions cancel, leaving the stack-parent coordinate positive. Therefore, on a valid nonterminated state,

$$
D _ { T , v } ^ { \mathrm { f } } = 2 \mathbf { 1 } \{ v \mathrm { ~ i s ~ a n ~ u n v i s i t e d ~ c h i l d ~ o f ~ } x \} , \qquad D _ { T , v } ^ { \mathrm { b } } = \mathbf { 1 } \{ v \mathrm { ~ i s ~ t h e ~ s t a c k ~ p a r e n t ~ o f ~ } x \} .\tag{71}
$$

The second indicator is zero at the root. Since attention is unmasked, this interpretation is asserted at the final query position; the scores at earlier positions also use the whole current prefix. All later layers preserve both score blocks.

Layers $3 , \ldots , k - 1$ : doubling the stored radius. Put $J = k - 3$ . For $j = 0 , \ldots , J - 1$ , layer $j + 3$ takes the state in Equation (68) with memories indexed by j and uses two heads:

<table><tr><td>head</td><td> $q _ { i }$ </td><td> $k _ { s }$ </td><td> $v _ { s }$ </td><td>output block</td></tr><tr><td>source propagation</td><td> $\overline { { S _ { i } ^ { ( j ) } } }$ </td><td> $U _ { s }$ </td><td> $\overline { { S _ { s } ^ { ( j ) } } }$ </td><td> $S _ { i } ^ { + }$ </td></tr><tr><td>child propagation</td><td> $\bar { C _ { i } ^ { ( j ) } }$ </td><td> $U _ { s }$ </td><td> $S _ { s } ^ { ( j ) }$ </td><td> $C _ { i } ^ { + }$ </td></tr></table>

Thus the physical and decoded updates are, respectively,

$$
\widehat { S } _ { i } ^ { ( j + 1 ) } = \widehat { S } _ { i } ^ { ( j ) } \oplus \sum _ { s } \langle S _ { i } ^ { ( j ) } , U _ { s } \rangle S _ { s } ^ { ( j ) } ,\tag{72}
$$

$$
\widehat { C } _ { i } ^ { ( j + 1 ) } = \widehat { C } _ { i } ^ { ( j ) } \oplus \sum _ { s } \langle C _ { i } ^ { ( j ) } , U _ { s } \rangle S _ { s } ^ { ( j ) } ,\tag{73}
$$

$$
S _ { e } ^ { ( j + 1 ) } = S _ { e } ^ { ( j ) } + \sum _ { f = ( a , b ) \in \mathcal { E } } S _ { e } ^ { ( j ) } [ a ] S _ { f } ^ { ( j ) } ,\tag{74}
$$

$$
C _ { e } ^ { ( j + 1 ) } = C _ { e } ^ { ( j ) } + \sum _ { f = ( a , b ) \in \mathcal { E } } C _ { e } ^ { ( j ) } [ a ] S _ { f } ^ { ( j ) } .\tag{75}
$$

Every update is a nonnegative head output added to a plus block. No block is overwritten and no Boolean threshold is applied. After j propagation layers, including $j = 0$ for the state after layer 2, the full state is

$$
\boxed { h _ { i } ^ { [ j + 2 ] } = ( I _ { i } , \widehat { S } _ { i } ^ { ( j ) } , \widehat { C } _ { i } ^ { ( j ) } , D _ { i } ^ { \mathrm { f } } , D _ { i } ^ { \mathrm { b } } , 0 , 0 ) , \qquad 0 \le j \le J . }\tag{76}
$$

To specify its meaning, define

$$
\mathcal { R } _ { r } ( w ) = \{ z : \mathrm { d i s t } _ { \mathcal { G } } ( w , z ) \leq r \} .
$$

The decoded memories have nonnegative integer coordinates and satisfy

$$
\mathrm { s u p p } _ { + } ( S _ { e } ^ { ( j ) } ) = \mathcal { R } _ { 2 ^ { j } } ( u ) , \qquad \mathrm { s u p p } _ { + } ( C _ { e } ^ { ( j ) } ) = \mathcal { R } _ { 2 ^ { j } } ( v ) , \qquad e = ( u , v ) .\tag{77}
$$

The base case follows directly from Equations (69) and (70). Self-loops may give an entry of two, which does not afect its positive support. For the induction step, write $r = 2 ^ { j }$ . Every summand in Equation (74) concatenates two paths of length at most r. Conversely, paths of length at most r remain through the residual. Split a path of length $r < \ell \leq 2 r$ after $r$ edges, at vertex a. Its remaining sufix is nonempty, so a has an outgoing edge and a corresponding source record $f .$ Both the coeficient at $a$ and the appropriate coordinate of $S _ { f } ^ { ( j ) }$ are positive, representing the whole path. The sum is over edge records, and multiple records with the same source only add positive contributions. This proves the invariant for both memories, including the zero-length path at a sink.

In particular, the input to the final layer is $h _ { i } ^ { [ k - 1 ] } = h _ { i } ^ { [ J + 2 ] }$ , with each edge’s child memory already representing radius $r = 2 ^ { J }$

Layer k: eligibility blocks and readout. Four parallel linear-attention heads read $h ^ { [ k - 1 ] }$
<table><tr><td>head</td><td> $q _ { i }$ </td><td> $k _ { s }$ </td><td> $v _ { s }$ </td><td>head output</td></tr><tr><td>goal reachability</td><td> $G _ { i }$ </td><td> $\overline { { C _ { s } ^ { ( J ) } } }$ </td><td> $V _ { s }$ </td><td> $F _ { i }$ </td></tr><tr><td>outgoing adjacency</td><td> $a _ { i }$ </td><td> $U _ { s }$ </td><td> $V _ { s }$ </td><td> $A _ { i }$ </td></tr><tr><td>visited counts</td><td> $\rho _ { i }$ </td><td> $\rho _ { s }$ </td><td> $a _ { s }$ </td><td>m</td></tr><tr><td>constant</td><td> $\rho _ { i }$ </td><td> $\zeta _ { s }$ </td><td> $\rho _ { s } \mathbf { 1 }$ </td><td>1</td></tr></table>

(78)

At $i = T$ , these outputs are

$$
F = \sum _ { e = ( u , v ) \in \mathcal { E } } C _ { e } ^ { ( J ) } [ g ] e _ { v } , \qquad A = \sum _ { e = ( u , v ) \in \mathcal { E } } \langle e _ { x } , e _ { u } \rangle e _ { v } , \qquad m = \sum _ { s } a _ { s } .
$$

The simple-graph assumption gives $A _ { v } \in \{ 0 , 1 \}$ , with $A _ { v } = 1$ exactly for an outgoing child. Also $m _ { v } = 0$ exactly for an unvisited vertex. For any outgoing child $v ,$ the support invariant gives

$$
F _ { v } > 0 \iff \ \mathrm { d i s t } _ { \mathcal { G } } ( v , g ) \leq r .
$$

This includes $v = g ,$ even $\operatorname { i f } g$ is a sink.

Choose an integer $B \geq 1$ bounding every coordinate of $F ,$ as specified below. The output projection writes only to the two fresh eligibility blocks, with preactivations $z _ { i }$ and $z _ { i } - \mathbf { 1 }$ , where

$$
z _ { i } = F _ { i } + B A _ { i } - 2 B m - B \mathbf { 1 } .
$$

Consequently, the complete state after the final layer is

$$
\begin{array} { r } { \left| { h } _ { i } ^ { [ k ] } = ( I _ { i } , \widehat { S } _ { i } ^ { ( J ) } , \widehat { C } _ { i } ^ { ( J ) } , D _ { i } ^ { \mathrm { f } } , D _ { i } ^ { \mathrm { b } } , \mathrm { R e L U } ( z _ { i } ) , \mathrm { R e L U } ( z _ { i } - \bf { 1 } ) ) . \right. } \end{array}\tag{79}
$$

At the query position define, by a linear readout,

$$
E _ { v } = Z _ { T , v } ^ { + } - Z _ { T , v } ^ { - } = \mathrm { R e L U } ( z _ { T , v } ) - \mathrm { R e L U } ( z _ { T , v } - 1 ) .\tag{80}
$$

All entries defining $z _ { T , v }$ are integers. An unvisited outgoing child with $F _ { v } > 0$ has $z _ { T , v } = F _ { v } \geq 1$ hence $E _ { v } = 1$ . A nonchild has $z _ { T , v } \le F _ { v } - B \le 0 ;$ a visited child has $z _ { T , v } \le F _ { v } + B - 2 B - B \le$ $F _ { v } - 2 B \le - B < 0 ;$ and a child with $F _ { v } = 0$ and $m _ { v } = 0$ has $z _ { T , v } = 0$ . Thus $E _ { v }$ is exactly the eligibility indicator.

Let $Q _ { N } = 2 ^ { \lceil \log _ { 2 } N \rceil }$ . The final linear readout assigns vertex v the logit

$$
\ell _ { v } = 4 ( Z _ { T , v } ^ { + } - Z _ { T , v } ^ { - } ) + D _ { T , v } ^ { \mathrm { f } } + D _ { T , v } ^ { \mathrm { b } } + \frac { N - v } { 4 Q _ { N } } \rho _ { T } .\tag{81}
$$

Assign each special token logit $- \rho _ { T }$ . The dyadic last term for vertex tokens lies in $[ 0 , 1 / 4 )$ and breaks ties by increasing label. Eligible children are preferred; if none exists, the forward DFS scores select an unvisited child, or the backtracking score selects the stack parent. This implements Theorem 3.7 with $L = r + 1 = 1 + 2 ^ { k - 3 }$

Precision and resources. Use a common fixed-point grid with step $2 ^ { - b }$ , where $b = O ( \log ( N + 1 ) )$ and suficient range to avoid overflow. First-layer head outputs are rounded to this grid before the integer-coeficient output projection. The two preactivations in each pair have the same nonconstant terms and biases difering by exactly one. The gate margins keep both active or both inactive as specified, so their physical outputs decode to an exact one or zero. The positional-selector and goal-selector parameters need only $O ( \log ( N + 1 ) )$ bits because the input length is polynomial in N. After this layer, decoded queries, keys, values, and memory increments are integers. An integer increment to the plus block preserves the exact decoded value.

After layer 2 the decoded memory entries are at most $P _ { 0 } = 2$ . Since $M \leq N ^ { 2 }$ , a uniform bound for the propagation is

$$
P _ { j + 1 } = ( N ^ { 2 } + 1 ) P _ { j } ^ { 2 } , \qquad P _ { j } = 2 ^ { 2 ^ { j } } ( N ^ { 2 } + 1 ) ^ { 2 ^ { j } - 1 } .
$$

Indeed, each coordinate increases by at most $M P _ { j } ^ { 2 }$ in addition to its old value $P _ { j }$ . Every destination has at most N incoming edges, so take $B = N \dot { P _ { J } }$ . Valid DFS traces have $O ( N )$ node occurrences, and all remaining intermediate values have magnitude at most a polynomial factor in N times B. Hence

$$
p = O { \Big ( } 2 ^ { k - 3 } \log ( N + 1 ) { \Big ) } = O _ { k } ( \log ( N + 1 ) )
$$

bits sufice, including fractional precision and the physical pairs’ bounded initial ofsets.

The fixed layout Equation (61) uses $O ( N )$ coordinates; the same two mutable memories are reused at every propagation layer. There are three heads in layer 1, four in layer 2, two in each

propagation layer, and four in the final layer. Zero heads may be added if a common head count is required. The layer count is

$$
\underbrace { 1 } _ { \mathrm { c o p i e s } } + \underbrace { 1 } _ { \mathrm { n e i g h b o r \ a g g r e g a t i o n \ a n d \ D F S } } + \underbrace { ( k - 3 ) } _ { \mathrm { d o u b l i n g s } } + \underbrace { 1 } _ { \mathrm { e l i g i b i l i t y } } = k .
$$

For $k = 3$ the propagation range is empty and the final layer reads the one-step child memories in $h ^ { [ 2 ] }$ □

Proof of Theorem 3.8. Apply Theorem B.2. Its constant head count, $O ( N )$ width, and $O _ { k } ( \log ( N + 1 ) )$ precision give representation size $O _ { k } ( N \log ( N + 1 ) )$ under the definition of Section 3. □

Distinguishing bounded reachability from distance minimization. The preceding construction tests whether each candidate child reaches the goal within one specified radius. It does not compare the distances of qualifying children. For example, with edges $u  v , u  g$ , and $v  g$ , horizon $L = 2$ , and v ordered before $^ { g , }$ both children qualify and the policy takes two moves although the target is one edge from u. Therefore the depth–horizon theorem alone does not imply the search-eficiency formula below. We state that formula for a stronger policy with an explicit distance-minimization requirement.

Definition B.3 (Distance-minimizing L-look-ahead DFS). At a nonterminated DFS state with current node $u ,$ goal ${ \mathit { g } } ,$ and visited set S, let $d _ { S } ( v , g )$ be the length of a shortest path from an unvisited child v to g all of whose vertices lie outside S, or $\infty$ if no such path exists. If some unvisited child satisfies $d _ { S } ( v , g ) \leq L - 1$ , select a child minimizing $d _ { S } ( v , g )$ , breaking ties by the fixed child order. Otherwise apply the ordinary DFS rule.

For $n \geq 2$ , define the worst-case eficiency of a fixed policy by

$$
\mathrm { e f f } = \operatorname* { m i n } _ { \mathcal { G } , n _ { s } , g } \frac { \mathrm { d i s t } _ { \mathcal { G } } ( n _ { s } , g ) } { \mathrm { n u m b e r ~ o f ~ C o T ~ m o v e s ~ b e f o r e ~ r e a c h i n g ~ } g } ,
$$

where the minimum is over instances with exactly n vertices reachable from $n _ { s }$ and a distinct reachable target $g .$

Lemma B.4 (Eficiency with distance-minimizing look-ahead). For every integer $L \ge 1$ , the distance-minimizing policy in Theorem B.3 has worst-case eficiency

$$
\mathrm { e f f } = { \left\{ \begin{array} { l l } { \displaystyle { \frac { L + 1 } { 2 n - L - 3 } } , } & { n \geq L + 2 , } \\ { 1 , } & { n < L + 2 . } \end{array} \right. }\tag{82}
$$

This statement is a property of that stronger policy and does not assert that the Boolean construction implements it.

Proof. Let $m = \mathrm { d i s t } _ { \mathcal { G } } ( n _ { s } , g )$ . If $m \leq L$ , the sufix after the root of a shortest source-to-goal path lies outside the initial visited set. The policy selects a child on a shortest residual path. The sufix of that path remains available after the move, and its minimum remaining length decreases by one. Induction shows that the target is reached in exactly m moves, giving eficiency one.

Suppose instead that $m \geq L + 1$ . Let D be the length of the final DFS-stack path to the target, and let T be the total number of moves. Every discovered tree edge outside that final path is traversed at most twice, while each of the D final-path edges is traversed once before termination. Since $D \geq m$ 2

$$
T \leq 2 ( n - 1 ) - D \leq 2 n - m - 2 .
$$

The function $m / ( 2 n - m - 2 )$ is increasing on the relevant range, so

$$
{ \frac { m } { T } } \geq { \frac { m } { 2 n - m - 2 } } \geq { \frac { L + 1 } { 2 n - L - 3 } } .
$$

If $n < L + 2$ , every reachable target has distance at most $n - 1 \leq L$ , so the first case applies to every instance.

For tightness when $n \geq L + 2$ , take a target path of length $L + 1$ from the root and a distracting chain containing the remaining $n - L - 2$ vertices, ordered before the target branch. The target is outside the horizon at the root and is unreachable from the distracting chain. The policy explores and backtracks along that chain before following the target path, taking

$$
2 ( n - L - 2 ) + ( L + 1 ) = 2 n - L - 3
$$

moves. The shortest-path distance is $L + 1$ , attaining the bound. When $n = L + 2$ , the distracting chain is empty and is omitted. □

Corollary B.5 (Conditional depth–eficiency implication). For every integer $k \geq 3$ , if a transformer implements the distance-minimizing policy of Theorem B.3 with $L = 1 + 2 ^ { k - 3 }$ , then its worst-case eficiency is

$$
\operatorname { e f f } = { \left\{ \begin{array} { l l } { \frac { 2 ^ { k - 3 } + 2 } { 2 n - 2 ^ { k - 3 } - 4 } , } & { n \geq 2 ^ { k - 3 } + 3 , } \\ { 1 , } & { n < 2 ^ { k - 3 } + 3 . } \end{array} \right. }\tag{83}
$$

The bounded-reachability transformer of Theorem 3.8 is not asserted to satisfy this additional premise. Proof. Substitute $L = 1 + 2 ^ { k - 3 }$ into Theorem B.4. □

## C Proof of Theorem 4.2

## C.1 Permutation symmetry and reduction to scalar parameters

Proof of Theorem $4 . 1 .$ Write $\theta = \left( A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 } \right)$ and, for $\pi \in S _ { \nu }$ , let $P _ { \pi } \in \mathbb { R } ^ { N \times N }$ be its permutation matrix, $P _ { \pi } e _ { v } = e _ { \pi ( v ) }$ . Define the action $\pi \cdot \theta : = ( P _ { \pi } A _ { 1 } P _ { \pi } ^ { T } , P _ { \pi } A _ { 2 } P _ { \pi } ^ { T } , P _ { \pi } B _ { 1 } P _ { \pi } ^ { T } , P _ { \pi } B _ { 2 } P _ { \pi } ^ { T } )$ ; it is linear and orthogonal for the Frobenius inner product.

Step 1: equivariance of the network. Relabeling the vertices of a prompt and a trace by π replaces every node one-hot $e _ { v } { \mathrm { \ b y } } P _ { \pi } e _ { v }$ and leaves the special tokens, the positions, and the length unchanged, so the input embeddings become ΠX with ${ \bf { I I } } = \mathrm { d i a g } ( P _ { \pi } , P _ { \pi } , P _ { \pi } , P _ { \pi } )$ on the four node blocks and the identity on the constant and positional coordinates. The fixed first layer (Theorem B.1) computes its attention scores from the positional coordinates only, its value maps copy node blocks or the constant vector ${ \mathbf { 1 } } _ { N } = P _ { \pi } { \mathbf { 1 } } _ { N }$ , and the ReLU and the residual act entrywise; hence its output transforms as ${ \pmb X } ^ { ( 2 ) } \mapsto \mathbf { \Pi } { \pmb X } ^ { ( 2 ) }$ (exactly, also with leakage $\delta > 0 )$ . In the second layer, writing ${ \pmb x } _ { j } = ( { \pmb a } _ { j } , { \pmb b } _ { j } , { \pmb c } _ { j } , { \pmb d } _ { j } )$ for the four node blocks,

$$
( \boldsymbol { \Pi } \pmb { x } _ { j } ) ^ { T } \boldsymbol { W } _ { K Q } ^ { ( 2 , 1 ) } [ \pi \cdot \theta ] \boldsymbol { \Pi } \pmb { x } _ { i } = a _ { j } ^ { T } P _ { \pi } ^ { T } P _ { \pi } \boldsymbol { A } _ { 1 } P _ { \pi } ^ { T } P _ { \pi } \boldsymbol { a } _ { i } + c _ { j } ^ { T } P _ { \pi } ^ { T } P _ { \pi } \boldsymbol { A } _ { 2 } P _ { \pi } ^ { T } P _ { \pi } \boldsymbol { a } _ { i } = x _ { j } ^ { T } \boldsymbol { W } _ { K Q } ^ { ( 2 , 1 ) } [ \theta ] \boldsymbol { x } _ { i } ,
$$

and likewise for the second head. The value maps satisfy ${ \pmb W } _ { V } ^ { ( 2 , h ) } { \pmb \Pi } { \pmb x } _ { j } = P _ { \pi } { \pmb W } _ { V } ^ { ( 2 , h ) } { \pmb x } _ { j }$ , the output matrix $\boldsymbol { W } _ { O } ^ { ( 2 ) }$ only routes blocks, ReL $\mathrm { U } ( P _ { \pi } y ) = P _ { \pi }$ ReLU(y), and the dynamic readout satisfies $\mathop { \mathrm { d i a g } } ( I _ { \pi ( n ) - 1 } , 0 , I _ { N - \pi ( n ) } ) = P _ { \pi } \mathop { \mathrm { d i a g } } ( I _ { n - 1 } , 0 , I _ { N - n } ) P _ { \pi } ^ { T }$ . Hence the vertex logits satisfy TF $( \boldsymbol { \Pi } \boldsymbol { X } ; \pi$ $\theta ) = P _ { \pi } \mathrm { T F } ( \boldsymbol { X } ; \theta )$ and $p _ { \pi \cdot \theta } ( \pi ( v ) ; \Pi X ) = p _ { \theta } ( v ; X )$ . Consequently, for every prompt, prefix, and trace,

$$
\ell _ { \pi ( \tau _ { \mathrm { P r o m p t } } ) , \pi ( \tau _ { \mathrm { p r e } } ) , \pi ( \tau _ { 0 } ) } ( \pi \cdot \theta ) = \ell _ { \tau _ { \mathrm { P r o m p t } } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } } ( \theta ) .\tag{84}
$$

Step 2: invariance of the averaged loss. Fix $\sigma \in S _ { \mathcal { V } }$ . Applying Equation (84) with $\pi = \sigma$ to the instance relabeled by $\sigma ^ { - 1 } \pi ^ { \prime }$ gives $\ell _ { \pi ^ { \prime } ( \cdot ) } ( \sigma \cdot \theta ) = \ell _ { \sigma ^ { - 1 } \pi ^ { \prime } ( \cdot ) } ( \theta )$ ; averaging over $\pi ^ { \prime } \in S _ { \mathcal { V } }$ and using that $\pi ^ { \prime } \mapsto \sigma ^ { - 1 } \pi ^ { \prime }$ is a bijection yields $\mathcal { L } _ { \mathrm { t r a i n } } ( \sigma \cdot \theta ) = \mathcal { L } _ { \mathrm { t r a i n } } ( \theta )$ for all θ and $\sigma .$

Step 3: the fixed subspace is invariant. A matrix commuting with all permutation matrices has constant diagonal and constant of-diagonal entries, because $S _ { N }$ acts transitively on $\{ ( i , i ) \}$ and on $\{ ( i , j ) : i \neq j \}$ . Hence the fixed-point set $\{ \theta : \sigma \cdot \theta = \theta \forall \sigma \}$ is exactly ${ \mathcal { F } } ,$ and $c { \pmb { E } } = c { \pmb { I } } + c { \pmb { E } } _ { \mathrm { o f f } } \in \mathcal { F }$ Wherever $\mathcal { L } _ { \mathrm { t r a i n } }$ is diferentiable, diferentiating $\mathcal { L } _ { \mathrm { t r a i n } } \circ \sigma = \mathcal { L } _ { \mathrm { t r a i n } }$ and using that the action is orthogonal gives $\nabla { \mathcal { L } } _ { \mathrm { t r a i n } } ( \sigma \cdot \theta ) = \sigma \cdot \nabla { \mathcal { L } } _ { \mathrm { t r a i n } } ( \theta )$ ; in particular $\nabla { \mathcal { L } } _ { \mathrm { t r a i n } } ( \theta ) \in { \mathcal { F } }$ for $\theta \in { \mathcal { F } }$ , so the vector field $- \alpha \nabla { \mathcal { L } } _ { \mathrm { t r a i n } }$ is tangent to $\mathcal { F }$ . At a nonsmooth point the same chain rule holds for the Clarke generalized gradient, $\partial \mathcal { L } _ { \mathrm { t r a i n } } ( \sigma \cdot \theta ) = \sigma \cdot \partial \mathcal { L } _ { \mathrm { t r a i n } } ( \theta )$ , so for $\theta \in { \mathcal { F } }$ the convex set $\partial \mathcal { L } _ { \mathrm { t r a i n } } ( \theta )$ is S<sub>V</sub>-invariant and therefore contains the group average of each of its elements; since the group average is the orthogonal projection $\Pi _ { \mathcal { F } }$ onto ${ \mathcal F } .$ , we get $\Pi _ { \mathcal { F } } \partial \mathcal { L } _ { \mathrm { t r a i n } } ( \theta ) \subset \partial \mathcal { L } _ { \mathrm { t r a i n } } ( \theta )$ . Together with the chain-rule inclusion $\partial ( \mathcal { L } _ { \mathrm { t r a i n } } | _ { \mathcal { F } } ) ( \theta ) \subset \Pi _ { \mathcal { F } } \partial \mathcal { L } _ { \mathrm { t r a i n } } ( \theta )$ for the linear embedding of ${ \mathcal { F } } _ { : }$ , every solution of the gradient-flow inclusion restricted to $\mathcal { F }$ is a solution of the full inclusion, and whenever $\mathcal { L } _ { \mathrm { t r a i n } }$ is diferentiable along the solution it is the unique one.

Step 4: reduction to one instance. For $\theta \in { \mathcal { F } }$ , Equation (84) gives $\ell _ { \pi ( \cdot ) } ( \theta ) = \ell _ { ( \cdot ) } ( \pi ^ { - 1 } \cdot \theta ) = \ell _ { ( \cdot ) } ( \theta )$ hence $\mathcal { L } _ { \mathrm { t r a i n } } ( \theta ) = \ell _ { \tau _ { \mathrm { P r o m p t } } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } } ( \theta )$ on $\mathcal { F }$ □

Since the weight matrices remain permutation invariant, we can transform the training on the whole matrices to training on the eight scalar parameters. Throughout the appendix,

$$
\ell ( a _ { 1 } , b _ { 1 } , a _ { 2 } , b _ { 2 } , g _ { 1 } , d _ { 1 } , g _ { 2 } , d _ { 2 } ) : = \ell _ { \mathrm { r e c o n p t } , \tau _ { \mathrm { P e } } , \tau _ { 0 } } \left( a _ { 1 } I + b _ { 1 } E _ { \mathrm { o f f } } , \ a _ { 2 } I + b _ { 2 } E _ { \mathrm { o f f } } , \ g _ { 1 } I + d _ { 1 } E _ { \mathrm { o f f } } , \ g _ { 2 } I + d _ { 2 } E _ { \mathrm { o f f } } \right)
$$

denotes the loss of the canonical instance viewed as a function of the eight scalars, and $\partial \ell / \partial a _ { i }$ etc.   
denote its ordinary partial derivatives, with the convention ReL $\mathrm { U } ^ { \prime } ( x ) = \mathbb { 1 } _ { x \geq 0 }$ at the kinks.

Lemma C.1. Let $( A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 } ) ( t )$ be a solution of the gradient flow $\dot { \theta } = - \alpha \nabla _ { \theta } \mathcal { L } _ { t r a i n }$ started in ${ \mathcal { F } } _ { : }$ , so that $A _ { 1 } = a _ { 1 } { \cal I } + b _ { 1 } { \cal E } _ { o f f } , \ \dots , \ B _ { 2 } = g _ { 2 } { \cal I } + d _ { 2 } { \cal E } _ { o f f }$ by Theorem $4 . 1 .$ . Then the scalars solve

$$
\dot { a } _ { i } = - \frac { \alpha } { N } \frac { \partial \ell } { \partial a _ { i } } , \qquad \dot { g } _ { i } = - \frac { \alpha } { N } \frac { \partial \ell } { \partial g _ { i } } ,\tag{85}
$$

$$
\dot { b } _ { i } = - \frac { \alpha } { N ( N - 1 ) } \frac { \partial \ell } { \partial b _ { i } } , \qquad \dot { d } _ { i } = - \frac { \alpha } { N ( N - 1 ) } \frac { \partial \ell } { \partial d _ { i } } , \qquad i \in \{ 1 , 2 \} ,\tag{86}
$$

that $i s ,$ the scalars follow the gradient flow of the single-instance loss ℓ with the learning rate normalization

$$
\alpha _ { a } = \alpha _ { g } = \frac { \alpha } { N } , \qquad \alpha _ { b } = \alpha _ { d } = \frac { \alpha } { N ( N - 1 ) } .\tag{87}
$$

Proof. By Step 4 of the proof of Theorem 4.1, $\mathcal { L } _ { \mathrm { t r a i n } } = \ell$ on ${ \mathcal { F } } ,$ and by Equation (84), ${ \mathcal { L } } _ { \mathrm { t r a i n } } ( \theta ) =$ $\begin{array} { r l } { \frac { 1 } { N ! } \sum _ { \pi } \ell _ { \tau _ { \mathrm { P r o m p t } } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } } ( \pi \cdot \theta ) } & { { } } \end{array}$ for every θ. Diferentiating at $\theta \in { \mathcal { F } }$ , where $\pi \cdot \theta = \theta$

$$
\nabla _ { A _ { 1 } } \mathcal { L } _ { \mathrm { t r a i n } } ( \theta ) = \frac { 1 } { N ! } \sum _ { \pi } P _ { \pi } ^ { T } G P _ { \pi } , \qquad G : = \nabla _ { A _ { 1 } } \ell _ { \mathrm { T p r o m p t } , \tau _ { \mathrm { p r e } } , \tau _ { 0 } } ( \theta ) ,
$$

and for any matrix G the conjugation average equals $\begin{array} { r } { \frac { \mathrm { T r } G } { N } I + \frac { \sum _ { k \neq l } G _ { k l } } { N ( N - 1 ) } E _ { \mathrm { o f f } } } \end{array}$ . By the chain rule, $\begin{array} { r } { \mathrm { T r } G = \sum _ { k } \partial \ell / \partial ( A _ { 1 } ) _ { k k } = \partial \ell / \partial a _ { 1 } } \end{array}$ and $\begin{array} { r } { \sum _ { k \neq l } G _ { k l } = \partial \ell / \partial b _ { 1 } } \end{array}$ . Hence

$$
\nabla _ { A _ { 1 } } \mathcal { L } _ { \mathrm { t r a i n } } = \frac { 1 } { N } \frac { \partial \ell } { \partial a _ { 1 } } \boldsymbol { I } + \frac { 1 } { N ( N - 1 ) } \frac { \partial \ell } { \partial b _ { 1 } } \boldsymbol { E } _ { \mathrm { o f f } } ,
$$

and likewise for $A _ { 2 } , B _ { 1 } , B _ { 2 }$ . Comparing the coeficients of I and $E _ { \mathrm { o f f } }$ in $\dot { A } _ { 1 } = - \alpha \nabla _ { A _ { 1 } } \mathcal { L } _ { \mathrm { t r a i n } }$ gives Equation (85). At a kink of the ReLU the same identity holds for every fixed permutationequivariant selection of the derivative, in particular for ReL $J ^ { \prime } ( 0 ) = 1$ □

To calculate the gradient with respect to each parameter, we need the formulation of the logits $\mathrm { T F } ( \mathbf { \boldsymbol { X } } , { \boldsymbol { \theta } } )$ . Since $\operatorname { T F } ( X , \theta )$ is the sum of the attention head values after ReLU, we first compute the node logits of each head separately, for a general directed graph and a general valid DFS prefix $( n _ { 0 } = n _ { \mathrm { R o o t } } , n _ { 1 } , . . . , n _ { t } )$ . We summarize the node logits in Table 1 and Table 2, separately.
<table><tr><td>Node type  $( v \neq n _ { t } )$ </td><td>Node logits</td></tr><tr><td>Unvisited child of  $n _ { t } \ ( m _ { v } = 0 )$ </td><td> $a _ { 2 } - b _ { 2 } + b _ { 2 } \cdot d _ { v } ^ { \mathrm { i n } }$  ,total</td></tr><tr><td>Visited child of  $n _ { t } \ ( m _ { v } \ge 1 )$  Non-child</td><td> $a _ { 2 } - b _ { 2 } + b _ { 2 } \cdot d _ { v } ^ { \mathrm { i n , t o t a l } } - b _ { 1 } m _ { v }$   $b _ { 2 } \cdot d _ { v } ^ { \mathrm { i n , t o t a l } } - b _ { 1 } m _ { v }$ </td></tr></table>

Table 1: Node logits of the forward head (pre-ReLU). A child of $n _ { t }$ is a vertex v with $( n _ { t } \to v ) \in \mathcal { E }$ $m _ { v }$ is the number of occurrences of v in $( n _ { \mathrm { R o o t } } , n _ { 1 } , . . . , n _ { t } )$ , i.e., in the CoT trace including the root token. $d _ { v } ^ { \mathrm { i n , t o t a l } }$ is the in-degree of v in the prompt’s edge list (self-loops included). The coordinate $v = n _ { t }$ is masked by the dynamic readout.

Derivation of the tables. Let the query be the last token, whose first block is $n _ { t } .$ . Since $A _ { 1 } n _ { t } =$ $a _ { 1 } n _ { t } + b _ { 1 } ( { \bf 1 } - n _ { t } )$ , and likewise for $A _ { 2 } , B _ { 1 } , B _ { 2 }$ , the only keys with nonzero score and nonzero value
<table><tr><td>Node type</td><td> $\mathrm { I f } \ v = n _ { \mathrm { R o o t } } ?$ </td><td>Node logits</td></tr><tr><td>Parent of  $n _ { t }$ </td><td>No</td><td> $g _ { 1 } - d _ { 1 }$ </td></tr><tr><td>Parent of  $n _ { t }$ </td><td>Yes</td><td> $g _ { 1 } - d _ { 1 } + d _ { 2 }$ </td></tr><tr><td>Tree child of  $n _ { t }$  (entered from  $n _ { t } )$ </td><td>No</td><td> $g _ { 1 } - g _ { 2 } - d _ { 1 } + d _ { 2 }$ </td></tr><tr><td> $n _ { \mathrm { R o o t } } .$  , neither parent nor  $n _ { t }$ </td><td>Yes</td><td> $d _ { 2 }$ </td></tr><tr><td>Any other  $v \neq n _ { t }$ </td><td>No</td><td>0</td></tr><tr><td> $n _ { t }$ </td><td>No</td><td> $k ( g _ { 2 } - g _ { 1 } + d _ { 1 } - d _ { 2 } ) - g _ { 1 } - d _ { 2 }$ </td></tr><tr><td> $n _ { t }$ </td><td> $\mathrm { Y e s }$ </td><td> $k ( g _ { 2 } - g _ { 1 } + d _ { 1 } - d _ { 2 } ) - g _ { 1 }$ </td></tr></table>

Table 2: Node logits of the backtracking head (pre-ReLU). k is the number of children of $n _ { t }$ that the DFS has entered from $n _ { t } ;$ since the search is currently at $n _ { t } ,$ each of these excursions has been completed. “Any other $v ^ { \flat }$ covers unvisited children, unvisited non-children, and visited vertices that are neither the parent, the root, nor a tree child of $n _ { t } .$ A visited out-neighbor of $n _ { t }$ that was reached by another route (the parent, the root, an ancestor, or a vertex finished in another branch) is not a tree child and falls into the parent, root, or “any other” row.

are, for the forward head, the edge-target tokens $( 0 , 0 , s _ { i } , t _ { i } )$ , with score $a _ { 2 }$ if $s _ { i } = n _ { t }$ and $b _ { 2 }$ otherwise and value $+ t _ { i } ,$ and the root and CoT tokens $( n _ { s } , \cdot , 0 , 0 ) , 0 \leq s \leq t ,$ with score $a _ { 1 }$ if $n _ { s } = n _ { t }$ and $b _ { 1 }$ otherwise and value $- n _ { s }$ . Hence, for $v \neq n _ { t }$ 2

$$
( \mathrm { h e a d _ { 1 } } ) _ { v } = ( a _ { 2 } - b _ { 2 } ) \mathbb { 1 } [ ( n _ { t }  v ) \in \mathcal { E } ] + b _ { 2 } d _ { v } ^ { \mathrm { i n , t o t a l } } - b _ { 1 } m _ { v } ,
$$

which is Table 1. For the backtracking head only the root token, with score $g _ { 1 } \mathbb { 1 } [ n _ { \mathrm { R o o t } } = n _ { t } ] +$ $d _ { 1 } \mathbb { 1 } [ n _ { \mathrm { R o o t } } \neq n _ { t } ]$ and value $- n _ { \mathrm { R o o t } }$ , and the CoT tokens $s \geq 1$ , with score $\sigma _ { s } = d _ { 1 } + d _ { 2 } + ( g _ { 1 } - d _ { 1 } ) \Im [ n _ { s } =$ $n _ { t } ] + ( g _ { 2 } - d _ { 2 } ) \mathbb { 1 } [ n _ { s - 1 } = n _ { t } ]$ and value $n _ { s - 1 } - n _ { s }$ , contribute. For $v \neq n _ { t }$ the $( d _ { 1 } + d _ { 2 } )$ -part telescopes to $( d _ { 1 } + d _ { 2 } ) \mathbb { 1 } [ v = n _ { \mathrm { R o o t } } ]$ , so

$$
( \mathrm { h e a d } _ { 2 } ) _ { v } = ( g _ { 1 } - d _ { 1 } ) A _ { v  n _ { t } } - ( g _ { 2 } - d _ { 2 } ) A _ { n _ { t }  v } + d _ { 2 } \mathbb { 1 } [ v = n _ { \mathrm { R o o t } } ] ,
$$

where $A _ { x  y }$ is the number of CoT steps $x  y$ . In a valid DFS prefix, $n _ { t }$ is entered once from its stack parent if $n _ { t } \neq n _ { \mathrm { R o o t } }$ , and otherwise only by the k returns from the children it has entered, each of which it left exactly once; it has not yet left towards its parent. Substituting gives the rows for $v \neq n _ { t }$ , every remaining vertex having value 0. For $v = n _ { t }$ , each arrival contributes $- ( g _ { 1 } + d _ { 2 } )$ , each departure $+ ( g _ { 2 } + d _ { 1 } )$ , and the root token − $- g _ { 1 }$ if $n _ { t } = n _ { \mathrm { R o o t } }$ , which gives the last two rows. □

By Theorem C.1, we only need to analyze the gradient of the single-instance loss ℓ on the canonical data $\mathcal { E } _ { 0 } , \tau _ { 0 }$ . Recall the initialization Equation (16): $a _ { 1 } = b _ { 1 } = a _ { 2 } = g _ { 1 } = g _ { 2 } = d _ { 2 } = c _ { 0 }$ and $b _ { 2 } = d _ { 1 } = c _ { 0 } ^ { \prime }$ with $0 < c _ { 0 } ^ { \prime } < c _ { 0 }$ . Throughout we write

$$
u : = a _ { 2 } - b _ { 1 } , \qquad v : = g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } , \qquad E : = e ^ { u _ { + } + v _ { + } } ,\tag{88}
$$

and we use the following region of parameter space:

$$
\begin{array} { r } { \mathcal { R } : = \big \{ b _ { 1 } > 0 , b _ { 2 } < b _ { 1 } , a _ { 2 } > 0 , g _ { 1 } - d _ { 1 } > 0 \big \} . } \end{array}\tag{89}
$$

The region is open on purpose: on the faces $a _ { 2 } = 0$ and $g _ { 1 } = d _ { 1 }$ the ReLUs $( a _ { 2 } ) _ { + }$ and $( g _ { 1 } - d _ { 1 } ) .$ <sub>+</sub> are at their kinks, where a Filippov solution may select any value in [0, 1] for their derivative, so the gradient formulas below hold on R but not on these faces.

Lemma C.2 (Scalar parameter gradient on canonical data). There is a constant $c _ { * } = c _ { * } ( K ) > 0$ such that the following holds for the flow Equation (85) started from Equation (16) with $0 < c _ { 0 } ^ { \prime } <$ $c _ { 0 } \leq c _ { * } N ^ { - 2 }$

1. (Invariant region.) The trajectory stays in R for all $t \geq 0$ . More precisely, $a _ { 1 } \equiv c _ { 0 } ; b _ { 1 }$ is non-decreasing and $b _ { 2 }$ is non-increasing with $0 \leq b _ { 2 } \leq c _ { 0 } ^ { \prime } ,$ so $b _ { 1 } \geq c _ { 0 } > 0$ and $b _ { 2 } < b _ { 1 } ;$ $u = a _ { 2 } - b _ { 1 } \geq 0$ , hence a<sub>2</sub> $\ge b _ { 1 } \ge c _ { 0 } > 0 ; g _ { 1 } - d _ { 1 } > 0 ; g _ { 2 } - d _ { 2 }$ is non-decreasing with $g _ { 2 } - d _ { 2 } \geq 0 .$ and $d _ { 2 }$ is non-increasing.

2. (Initial phase.) With $\begin{array} { r } { T _ { 0 } : = \frac { N } { 4 ( K + 1 ) \alpha } } \end{array}$ , we have $b _ { 2 } ( t ) = 0$ and $d _ { 2 } ( t ) \leq 0$ for all $t \geq T _ { 0 }$

3. (Gradient after the initial phase.) For $t \geq T _ { 0 }$ ，

$$
\frac { \partial \ell } { \partial a _ { 1 } } = 0 ,\tag{90}
$$

$$
\frac { \partial \ell } { \partial a _ { 2 } } = - \frac { e ^ { g _ { 1 } - d _ { 1 } } + N - 2 } { D _ { 1 } } - \sum _ { i = 1 } ^ { K } \frac { e ^ { g _ { 1 } - d _ { 1 } } + N - K - 1 + ( i - 1 ) E \mathbb { 1 } _ { u < 0 } } { D _ { 2 , i } } + \frac { K E \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } ,\tag{91}
$$

$$
\frac { \partial \ell } { \partial b _ { 1 } } = - \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { u \geq 0 } } { D _ { 2 , i } } - \frac { K E \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } ,\tag{92}
$$

$$
b _ { 2 } \equiv 0 ( s o \ t h e \ s e l e c t e d \ v a l u e \ o f \ ) \ d \ell / \partial b _ { 2 } \ a l o n g \ t h e \ f i o w \ i s \ 0 ) , ( b _ { 2 } ) _ { + } = ( b _ { 2 } - b _ { 1 } ) _ { + } = 0 ,\tag{93}
$$

$$
\frac { \partial \ell } { \partial g _ { 1 } } = \frac { e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { v \geq 0 } + e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 2 , i } } - \frac { K ( N - 1 ) } { D _ { 3 } } - \frac { N - K - 1 + K E \mathbb { 1 } _ { v < 0 } } { D _ { 4 } } ,\tag{94}
$$

$$
\frac { \partial \ell } { \partial g _ { 2 } } = - \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { v \geq 0 } } { D _ { 2 , i } } - \frac { K E \mathbb { 1 } _ { v \geq 0 } } { D _ { 4 } } ,\tag{95}
$$

$$
\frac { \partial \ell } { \partial d _ { 1 } } = - \frac { \partial \ell } { \partial g _ { 1 } } , \qquad \frac { \partial \ell } { \partial d _ { 2 } } = - \frac { \partial \ell } { \partial g _ { 2 } } w h e r e v e r d _ { 2 } < 0 ,\tag{96}
$$

where the denominators are

$$
D _ { 1 } = e ^ { a _ { 2 } } + e ^ { g _ { 1 } - d _ { 1 } } + N - 2 ,\tag{97}
$$

$$
D _ { 2 , i } = ( K - i + 1 ) e ^ { a _ { 2 } } + ( i - 1 ) E + e ^ { g _ { 1 } - d _ { 1 } } + N - K - 1 ,\tag{98}
$$

$$
D _ { 3 } = e ^ { g _ { 1 } - d _ { 1 } } + N - 1 ,\tag{99}
$$

$$
D _ { 4 } = K E + e ^ { g _ { 1 } - d _ { 1 } } + N - K - 1 .\tag{100}
$$

Note that the relation $\partial \ell / \partial d _ { 1 } = - \partial \ell / \partial g _ { 1 }$ holds for all $t \geq 0$ , because by Table 2 every unmasked logit depends on $( g _ { 1 } , d _ { 1 } )$ only through $g _ { 1 } - d _ { 1 }$ ; the relation for $( g _ { 2 } , d _ { 2 } )$ holds wherever $d _ { 2 } < 0 ;$ at times $t \geq T _ { 0 }$ with $d _ { 2 } = 0$ it can fail, but $g _ { 2 } + ( N - 1 ) d _ { 2 }$ is still conserved (Step 5 of the proof).

Proof. For the cross-entropy loss $- \log ( p _ { \theta } ( v ; X ) )$ , the gradient with respect to a parameter θ is

$$
\frac { \partial ( - \log ( p _ { \theta } ( v ; \mathbf { X } ) ) ) } { \partial \theta } = - \frac { \partial \mathrm { T F } ( \mathbf { X } , \theta ) _ { v } } { \partial \theta } + \sum _ { v ^ { \prime } \in \mathcal { V } } p _ { \theta } ( v ^ { \prime } ; \mathbf { \boldsymbol { X } } ) \frac { \partial \mathrm { T F } ( \mathbf { \boldsymbol { X } } , \theta ) _ { v ^ { \prime } } } { \partial \theta } ,\tag{101}
$$

and the gradient of ℓ is the sum of Equation (101) over the $2 K + 2$ steps of $\tau _ { 0 }$ .

Step 1: node logits along the canonical trace. In the canonical trace, DFS starts from node $N - 3$ and moves forward to 1. After exploring all K child nodes of 1, it then returns to $N - 3$ . Therefore, there are four types of steps in the trajectory. We next compute the node logits for these four steps from Tables 1 and 2; all nodes $2 , \ldots , K + 1$ have in-degree one, and the nodes $N - 2 , N - 3 ,$ 1 have in-degree one as well. Recall that the logit of the current node $n _ { t }$ is masked to 0, and that the remaining $N - K - 4$ vertices are isolated and unvisited, so their logits are 0. In the following, we use $p ( v )$ as an abbreviation of $p _ { \theta } ( v ; X )$

Case 1: $N - 3  1$ . The logits $\mathrm { T F } ( \pmb { X } , \theta )$ of all nodes are

• v = 1: TF(X, θ)<sub>v</sub> = (a<sub>2</sub>)<sub>+</sub>

$v = N - 2$ (parent of N − 3): TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub> + (g<sub>1</sub> − d<sub>1</sub>)<sub>+</sub>

• v = N − 1 (root): TF(X, θ)<sub>v</sub> = (−b<sub>1</sub>)<sub>+</sub> + (d<sub>2</sub>)<sub>+</sub>

• 2 ≤ v ≤ K + 1: TF(X, θ)<sub>v</sub> = (b<sub>2</sub>)<sub>+</sub>

• Other: $\mathrm { T F } ( \boldsymbol { X } , \boldsymbol { \theta } ) _ { v } = 0$

Case 2: 1 → i for $2 \leq i \leq K + 1$ . The logits $\mathrm { T F } ( \mathbf { \boldsymbol { X } } , { \boldsymbol { \theta } } )$ of all nodes are

$i \le v \le K + 1$ (unvisited children): TF(X, θ)<sub>v</sub> = (a<sub>2</sub>)<sub>+</sub>.

$2 \leq v \leq i - 1$ (tree children, each visited once): $\mathrm { T F } ( \boldsymbol { X } , \theta ) _ { v } = ( a _ { 2 } - b _ { 1 } ) _ { + } + ( g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } ) _ { + }$

• v = N − 3 (parent) : TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub> + (g<sub>1</sub> − d<sub>1</sub>)<sub>+</sub>.

• v = N − 2: TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub>.

$v = N - 1 \colon \mathrm { T F } ( X , \theta ) _ { v } = ( - b _ { 1 } ) _ { + } + ( d _ { 2 } ) _ { + }$

• Other: $\mathrm { T F } ( \boldsymbol { X } , \boldsymbol { \theta } ) _ { v } = 0$

Case 3: i → 1 for $2 \leq i \leq K + 1$ . At this step node 1 has occurred i − 1 times in the trace. The logits $\mathrm { T F } ( \pmb { X } , \theta )$ of all nodes are

• v = 1 (parent): TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − (i − 1)b<sub>1</sub>)<sub>+</sub> + (g<sub>1</sub> − d<sub>1</sub>)<sub>+</sub>.

$v = N - 2 \mathrm { ~ o r ~ } N - 3 \colon \mathrm { T F } ( X , \theta ) _ { v } = ( b _ { 2 } - b _ { 1 } ) _ { + }$

• v = N − 1: TF(X, θ)<sub>v</sub> = (−b<sub>1</sub>)<sub>+</sub> + (d<sub>2</sub>)<sub>+</sub>

• 2 ≤ v ≤ i − 1: TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub>.

$i + 1 \le v \le K + 1 \colon \operatorname { T F } ( \boldsymbol { X } , \boldsymbol { \theta } ) _ { v } = ( b _ { 2 } ) _ { + } .$

• Other: TF(X, θ)<sub>v</sub> = 0

Case 4: 1 → N − 3. The logits TF(X, θ) of all nodes are

• v = N − 3 (parent): TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub> + (g<sub>1</sub> − d<sub>1</sub>)<sub>+</sub>

• v = N − 2: TF(X, θ)<sub>v</sub> = (b<sub>2</sub> − b<sub>1</sub>)<sub>+</sub>

• v = N − 1: TF(X, θ)<sub>v</sub> = (−b<sub>1</sub>)<sub>+</sub> + (d<sub>2</sub>)<sub>+</sub>

$2 \leq v \leq K + 1$ (tree children): TF(X, θ)<sub>v</sub> = (a<sub>2</sub> − b<sub>1</sub>)<sub>+</sub> + (g<sub>1</sub> − d<sub>1</sub> − g<sub>2</sub> + d<sub>2</sub>)<sub>+</sub>

• Other: TF(X, θ)<sub>v</sub> = 0

In the gradient formulas below we index the K steps of Case 2 and of Case 3 by $i \in [ K ]$ , i.e., we shift the child label by one, so that at the i-th such step there are $K - i + 1$ unvisited and $i - 1$ visited children of node 1.

Step 2: the gradient on R. On the region R of Equation (89) we have $b _ { 2 } - b _ { 1 } < 0 , b _ { 2 } - ( i - 1 ) b _ { 1 } \leq$ $b _ { 2 } - b _ { 1 } < 0$ and $- b _ { 1 } < 0$ , so the terms $( b _ { 2 } - b _ { 1 } ) _ { + } , ( b _ { 2 } - ( i - 1 ) b _ { 1 } ) _ { + } , ( - b _ { 1 } ) _ { + }$ and their derivatives vanish identically, while $( a _ { 2 } ) _ { + } = a _ { 2 }$ and $( g _ { 1 } - d _ { 1 } ) _ { + } = g _ { 1 } - d _ { 1 }$ with derivative 1, because $a _ { 2 } > 0$ and $g _ { 1 } - d _ { 1 } > 0$ on the open region R. The remaining kinks $u = 0 , v = 0 , b _ { 2 } = 0$ and $d _ { 2 } = 0$ can lie in R; there the indicators $\mathbb { 1 } _ { u \ge 0 } , \mathbb { 1 } _ { v \ge 0 } , \mathbb { 1 } _ { b _ { 2 } \ge 0 } , \mathbb { 1 } _ { d _ { 2 } \ge 0 }$ below stand for the values in [0, 1] selected by the Filippov solution (the weights of the convex combination of the one-sided fields). Applying Equation (101) to the four cases gives, on $\mathcal { R }$

$$
\frac { \partial \ell } { \partial a _ { 1 } } = 0 ,\tag{102}
$$

$$
\frac { \partial \ell } { \partial a _ { 2 } } = - 1 + \frac { e ^ { a _ { 2 } } } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \Bigl ( - 1 + \frac { ( K - i + 1 ) e ^ { a _ { 2 } } + ( i - 1 ) E \mathbb { 1 } _ { u \geq 0 } } { D _ { 2 , i } } \Bigr ) + \frac { K E \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } ,\tag{103}
$$

$$
\frac { \partial \ell } { \partial b _ { 1 } } = - \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { u \geq 0 } } { D _ { 2 , i } } - \frac { K E \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } ,\tag{104}
$$

$$
\frac { \partial \ell } { \partial b _ { 2 } } = e ^ { ( b _ { 2 } ) _ { + } } \mathbb { 1 } _ { b _ { 2 } \geq 0 } \Big [ \frac { K } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { K - i } { D _ { 3 , i } } \Big ] ,\tag{105}
$$

$$
\frac { \partial \ell } { \partial g _ { 1 } } = \frac { e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { v \geq 0 } + e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 2 , i } } + \sum _ { i = 1 } ^ { K } \Bigl ( - 1 + \frac { e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 3 , i } } \Bigr ) - 1 + \frac { K E \mathbb { 1 } _ { v \geq 0 } + e ^ { g _ { 1 } - d _ { 1 } } } { D _ { 4 } } ,\tag{106}
$$

$$
\frac { \partial \ell } { \partial g _ { 2 } } = - \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) E \mathbb { 1 } _ { v \geq 0 } } { D _ { 2 , i } } - \frac { K E \mathbb { 1 } _ { v \geq 0 } } { D _ { 4 } } ,\tag{107}
$$

$$
\frac { \partial \ell } { \partial d _ { 1 } } = - \frac { \partial \ell } { \partial g _ { 1 } } ,\tag{108}
$$

$$
\frac { \partial \ell } { \partial d _ { 2 } } = - \frac { \partial \ell } { \partial g _ { 2 } } + e ^ { ( d _ { 2 } ) _ { + } } \mathbb { 1 } _ { d _ { 2 } \geq 0 } \left[ \frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 3 , i } } + \frac { 1 } { D _ { 4 } } \right] ,\tag{109}
$$

where the denominators are the softmax normalizers of the four cases,

$$
D _ { 1 } = e ^ { a _ { 2 } } + e ^ { g _ { 1 } - d _ { 1 } } + e ^ { ( d _ { 2 } ) _ { + } } + K e ^ { ( b _ { 2 } ) _ { + } } + N - K - 3 ,\tag{110}
$$

$$
D _ { 2 , i } = ( K - i + 1 ) e ^ { a _ { 2 } } + ( i - 1 ) E + e ^ { g _ { 1 } - d _ { 1 } } + 1 + e ^ { ( d _ { 2 } ) _ { + } } + N - K - 3 ,\tag{111}
$$

$$
D _ { 3 , i } = e ^ { g _ { 1 } - d _ { 1 } } + 2 + e ^ { ( d _ { 2 } ) _ { + } } + ( i - 1 ) + ( K - i ) e ^ { ( b _ { 2 } ) _ { + } } + N - K - 3 ,\tag{112}
$$

$$
D _ { 4 } = K E + e ^ { g _ { 1 } - d _ { 1 } } + 1 + e ^ { ( d _ { 2 } ) _ { + } } + N - K - 3 .\tag{113}
$$

Each denominator has exactly N terms, one per vertex token, including the masked current node with weight $e ^ { 0 } = 1$ . In particular $D _ { 1 } , D _ { 2 , i } , D _ { 3 , i } , D _ { 4 } \geq N$ on $\mathcal { R }$ , and each of the $2 K + 2$ steps contributes a quantity in $[ - 1 , 1 ]$ to every partial derivative, so that

$$
\left| \frac { \partial \ell } { \partial \theta } \right| \leq 2 K + 2 \qquad \mathrm { o n } ~ \mathcal { R } , ~ \mathrm { f o r ~ e v e r y ~ s c a l a r } ~ \theta .\tag{114}
$$

The identity Equation (108) holds on all of parameter space, since $g _ { 1 } , d _ { 1 }$ enter the unmasked logits only through $g _ { 1 } - d _ { 1 } ;$ ; the extra term in Equation (109) comes from the logit $( d _ { 2 } ) _ { + }$ of the root in every step.

Step 3: R is invariant. The initial point lies in R: $b _ { 1 } = c _ { 0 } > 0 , b _ { 2 } = c _ { 0 } ^ { \prime } < c _ { 0 } = b _ { 1 } , a _ { 2 } = c _ { 0 } > 0$ $g _ { 1 } - d _ { 1 } = c _ { 0 } - c _ { 0 } ^ { \prime } > 0$ . As long as the trajectory is in R, Equations (102), (104), (105) and (107) and Equation (85) give $\dot { a } _ { 1 } = 0 , \dot { b } _ { 1 } \geq 0 , \dot { b } _ { 2 } \leq 0$ and ${ \dot { g } } _ { 2 } \geq 0 ;$ ; moreover $\begin{array} { r } { \dot { d } _ { 2 } = - \frac { \alpha } { N ( N - 1 ) } \frac { \partial \ell } { \partial d _ { 2 } } \le 0 } \end{array}$ because both terms of Equation (109) are nonnegative, and

$$
\frac { \mathrm { d } } { \mathrm { d } t } ( g _ { 2 } - d _ { 2 } ) = - \frac { \alpha } { N - 1 } \frac { \partial \ell } { \partial g _ { 2 } } + \frac { \alpha } { N ( N - 1 ) } e ^ { ( d _ { 2 } ) _ { + } } \mathbb { 1 } _ { d _ { 2 } \geq 0 } \Big [ \frac { 1 } { D _ { 1 } } + \sum _ { i } \frac { 1 } { D _ { 2 , i } } + \sum _ { i } \frac { 1 } { D _ { 3 , i } } + \frac { 1 } { D _ { 4 } } \Big ] \geq 0 ,\tag{115}
$$

with strict inequality whenever $v \geq 0$ , because then $- \partial \ell / \partial g _ { 2 } \ge K E / D _ { 4 } > 0$ . Hence $b _ { 1 } \ \geq \ c _ { 0 }$ $b _ { 2 } - b _ { 1 }$ is non-increasing and stays negative, and $g _ { 2 } \mathrm { ~ - ~ } d _ { 2 } \ge \mathrm { ~ 0 ~ }$ is non-decreasing. Moreover $b _ { 2 }$ never becomes negative: on $\{ b _ { 2 } < 0 \} \cap \mathcal { R }$ the field Equation (105) vanishes, so if $b _ { 2 } ( t _ { 3 } ) < 0$ and $t ^ { \prime } = \operatorname* { s u p } \{ t \leq t _ { 3 } : b _ { 2 } ( t ) \geq 0 \}$ , then $b _ { 2 }$ is constant on $( t ^ { \prime } , t _ { 3 } ]$ and $b _ { 2 } ( t _ { 3 } ) = b _ { 2 } ( t ^ { \prime } ) = 0$ by continuity, a contradiction; hence $0 \leq b _ { 2 } \leq c _ { 0 } ^ { \prime }$ . Since $\mathcal { R }$ is open and these inequalities keep $b _ { 1 } > 0$ and $b _ { 2 } < b _ { 1 }$ , a continuous trajectory can leave R only through one of the two faces $\{ a _ { 2 } = 0 \}$ or $\{ g _ { 1 } - d _ { 1 } = 0 \}$ . We show that neither face is reached. The arguments use the field only on $\mathcal { R } .$ in a neighborhood of the faces, and never on the faces themselves, where the selections for $( a _ { 2 } ) _ { + }$ and $( g _ { 1 } - d _ { 1 } ) _ { + }$ are free.

The face $a _ { 2 } = 0$ . On the open set $\{ u < 0 \} \cap \mathcal { R }$ we have $\mathbb { 1 } _ { u \geq 0 } = 0$ , so by Equations (103) and (104)

$$
\frac { \partial \ell } { \partial a _ { 2 } } = - \bigg ( 1 - \frac { e ^ { a _ { 2 } } } { D _ { 1 } } \bigg ) - \sum _ { i = 1 } ^ { K } \bigg ( 1 - \frac { ( K - i + 1 ) e ^ { a _ { 2 } } } { D _ { 2 , i } } \bigg ) < 0 , \qquad \frac { \partial \ell } { \partial b _ { 1 } } = 0 ,
$$

because $D _ { 1 } > e ^ { a _ { 2 } }$ and $D _ { 2 , i } > ( K - i + 1 ) e ^ { a _ { 2 } }$ . Hence $u ^ { \prime } = \dot { a } _ { 2 } - \dot { b } _ { 1 } > 0$ on $\{ u < 0 \} \cap \mathcal { R }$ ; since $u ( 0 ) = 0$ the trajectory never enters $\{ u < 0 \}$ , so $u \geq 0$ and $a _ { 2 } \geq b _ { 1 } \geq c _ { 0 } > 0$ for all t. In particular the face $a _ { 2 } = 0$ is never reached.

The face $g _ { 1 } - d _ { 1 } = 0$ . Since $v ( 0 ) = g _ { 1 } ( 0 ) - d _ { 1 } ( 0 ) - ( g _ { 2 } - d _ { 2 } ) ( 0 ) = c _ { 0 } - c _ { 0 } ^ { \prime } > 0$ , by continuity $v > 0$ on some interval $[ 0 , \delta )$ , where Equation (115) gives $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \big ( g _ { 2 } - d _ { 2 } \big ) \geq \frac { \alpha } { N - 1 } \frac { K E } { D _ { 4 } } > 0 ; } \end{array}$ ; as $g _ { 2 } - d _ { 2 }$ is non-decreasing, $( g _ { 2 } - d _ { 2 } ) ( t ) > 0$ for all $t > 0$ . Fix $a \in ( 0 , \delta )$ so small that $g _ { 1 } - d _ { 1 } > 0$ on $[ 0 , a ]$ 2 and let $\epsilon _ { 0 } : =$ min $[ ( g _ { 2 } - d _ { 2 } ) ( a )$ , log $2 \} > 0$ . For $t \geq a$ , at every state of R with $g _ { 1 } - d _ { 1 } < \epsilon _ { 0 }$ we have $v = ( g _ { 1 } - d _ { 1 } ) - ( g _ { 2 } - d _ { 2 } ) < \epsilon _ { 0 } - ( g _ { 2 } - d _ { 2 } ) ( a ) \leq 0 \quad$ , so $\mathbb { 1 } _ { v \ge 0 } = 0$ in a neighborhood of the state, and with $e ^ { g _ { 1 } - d _ { 1 } } \leq 2$ and all denominators at least N, Equation (106) gives

$$
\frac { \partial \ell } { \partial g _ { 1 } } \le \frac { 2 } { N } + \frac { 2 K } { N } - K \Big ( 1 - \frac { 2 } { N } \Big ) - \Big ( 1 - \frac { 2 } { N } \Big ) = ( K + 1 ) \Big ( \frac { 4 } { N } - 1 \Big ) < 0 .\tag{116}
$$

Hence $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } ( g _ { 1 } - d _ { 1 } ) = - \frac { \alpha } { N } \frac { \partial \ell } { \partial g _ { 1 } } + \frac { \alpha } { N ( N - 1 ) } \frac { \partial \ell } { \partial d _ { 1 } } = - \frac { \alpha } { N - 1 } \frac { \partial \ell } { \partial g _ { 1 } } > 0 } \end{array}$ on the whole set $\{ g _ { 1 } - d _ { 1 } < \epsilon _ { 0 } \} \cap \mathcal { R }$ (for $t \geq a )$ , not only at a single time, and the one-sided barrier argument gives $g _ { 1 } - d _ { 1 } \geq \operatorname* { m i n } \{ ( g _ { 1 } -$ $d _ { 1 } ) ( a ) , \epsilon _ { 0 } \} > 0$ for all $t \geq a$ . Thus $g _ { 1 } - d _ { 1 } > 0$ for all $t > 0$ . This proves item 1.

Step 4: the initial phase. By Equation (114) and Equation (85), every scalar moves at speed at most $( 2 K + 2 ) \alpha / N$ , so on $[ 0 , T _ { 0 } ]$ with $\begin{array} { r } { T _ { 0 } = \frac { N } { 4 ( K + 1 ) \alpha } } \end{array}$ every parameter changes by at most $\scriptstyle { \frac { 1 } { 2 } } ;$ since $0 < c _ { 0 } ^ { \prime } < c _ { 0 } \le c _ { * } N ^ { - 2 } < { \frac { 1 } { 4 } }$ (by the definition of $c _ { * } \ \mathrm { b e l o w } )$ , all parameters lie in $[ - \frac { 1 } { 2 } , 1 ]$ on $[ 0 , T _ { 0 } ]$ Consequently every exponent appearing in the denominators is bounded by an absolute constant, and there exist absolute constants $C _ { 1 } , C _ { 2 } \geq 0$ such that

$$
\operatorname* { m a x } _ { i } \{ D _ { 1 } , D _ { 2 , i } , D _ { 3 , i } , D _ { 4 } \} \leq N + C _ { 1 } K + C _ { 2 } \qquad \mathrm { o n ~ } [ 0 , T _ { 0 } ] .\tag{117}
$$

While $b _ { 2 } > 0$ (so that $\mathbb { 1 } _ { b _ { 2 } \geq 0 } = 1 )$ , Equation (105) gives

$$
\frac { \partial \ell } { \partial b _ { 2 } } \geq \frac { K + \frac { K ( K - 1 ) } { 2 } } { N + C _ { 1 } K + C _ { 2 } } \geq \frac { K } { N + C _ { 1 } K + C _ { 2 } } , \qquad \mathrm { s o } \qquad \dot { b } _ { 2 } \leq - \frac { \alpha } { N ( N - 1 ) } \frac { K } { N + C _ { 1 } K + C _ { 2 } } ,\tag{118}
$$

and while $d _ { 2 } > 0$ , Equation (109) gives

$$
\frac { \partial \ell } { \partial d _ { 2 } } \geq \frac { 2 K + 2 } { N + C _ { 1 } K + C _ { 2 } } \geq \frac { K } { N + C _ { 1 } K + C _ { 2 } } , \qquad \mathrm { s o } \qquad \dot { d } _ { 2 } \leq - \frac { \alpha } { N ( N - 1 ) } \frac { K } { N + C _ { 1 } K + C _ { 2 } } .\tag{119}
$$

Over the time interval $[ 0 , T _ { 0 } ]$ each of $b _ { 2 } , d _ { 2 }$ therefore decreases, as long as it is positive, by at least

$$
\begin{array} { r } { \frac { \alpha } { N ( N - 1 ) } \displaystyle \frac { K } { N + C _ { 1 } K + C _ { 2 } } \frac { N } { 4 ( K + 1 ) \alpha } = \frac { K } { 4 ( K + 1 ) ( N - 1 ) ( N + C _ { 1 } K + C _ { 2 } ) } } \\ { \geq \frac { K N ^ { - 2 } } { 4 ( K + 1 ) ( 1 + C _ { 1 } K + C _ { 2 } ) } = : c _ { * } N ^ { - 2 } , } \end{array}\tag{120}
$$

where we used $( N - 1 ) ( N + C _ { 1 } K + C _ { 2 } ) \le N ^ { 2 } ( 1 + C _ { 1 } K + C _ { 2 } )$ . Since $b _ { 2 } ( 0 ) = c _ { 0 } ^ { \prime } < c _ { 0 } \leq c _ { * } N ^ { - 2 }$ and $d _ { 2 } ( 0 ) = c _ { 0 } \le c _ { * } N ^ { - 2 }$ , both $b _ { 2 }$ and $d _ { 2 }$ reach 0 before time $T _ { 0 }$ . Since $b _ { 2 }$ is non-increasing and $b _ { 2 } \geq 0$ $( \mathrm { S t e p ~ 3 } ) , b _ { 2 } = 0$ for all $t \geq T _ { 0 } ;$ since $d _ { 2 }$ is non-increasing, $d _ { 2 } \leq 0$ for all $t \geq T _ { 0 }$ . This proves item 2. Step 5: simplification. For $t \geq T _ { 0 }$ we have $b _ { 2 } = 0 , e ^ { ( b _ { 2 } ) _ { + } } = e ^ { ( d _ { 2 } ) _ { + } } = 1$ , and $\Im { _ { d _ { 2 } \geq 0 } } = 0$ wherever $d _ { 2 } < 0$ . The term with $\mathbb { 1 } _ { d _ { 2 } \geq 0 }$ in Equation (109) enters only ${ \dot { d } } _ { 2 }$ and does not afect the reduced system: the set $\{ t \ge T _ { 0 } : d _ { 2 } ( t ) = 0 \}$ is an interval, as $d _ { 2 } \leq 0$ is non-increasing; on its interior $\dot { d } _ { 2 } = 0$ , so both nonnegative terms of $\partial \ell / \partial d _ { 2 }$ in Equation (109) vanish, and after it $d _ { 2 } < 0$ . In both cases $\begin{array} { r } { \dot { g } _ { 2 } + ( N - 1 ) \dot { d } _ { 2 } = - \frac { \alpha } { N } \big ( \frac { \partial \ell } { \partial q _ { 2 } } + \frac { \partial \dot { \ell } } { \partial d _ { 2 } } \big ) = 0 } \end{array}$ a.e., so $g _ { 2 } + ( N - 1 ) d _ { 2 }$ is constant on $[ T _ { 0 } , \infty )$ and $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } ( g _ { 2 } - d _ { 2 } ) = \frac { N } { N - 1 } \dot { g } _ { 2 } = - \frac { \alpha } { N - 1 } \frac { \partial \ell } { \partial g _ { 2 } } } \end{array}$ for ${ \mathrm { a . e . ~ } } t \geq T _ { 0 }$ . Substituting into Equations (103), (104) and (106) to (109) and using

$$
\begin{array} { c } { { - 1 + \displaystyle \frac { e ^ { \alpha _ { 2 } } } { D _ { 1 } } = - \displaystyle \frac { e ^ { \sigma _ { 1 } - d _ { 1 } } + N - 2 } { D _ { 1 } } , } } \\ { { - 1 + \displaystyle \frac { ( K - i + 1 ) e ^ { \alpha _ { 2 } } + ( i - 1 ) E \mathbb { 1 } _ { u \geq 0 } } { D _ { 2 , i } } = - \displaystyle \frac { e ^ { \sigma _ { 1 } - d _ { 1 } } + N - K - 1 } { D _ { 2 , i } } - \displaystyle \frac { ( i - 1 ) E \mathbb { 1 } _ { u \leq 0 } } { D _ { 2 , i } } , } } \\ { { \displaystyle \sum _ { i = 1 } ^ { K } \Bigl ( - 1 + \displaystyle \frac { e ^ { \sigma _ { 1 } - d _ { 1 } } } { D _ { 3 , i } } \Bigr ) = - \displaystyle \frac { K ( N - 1 ) } { D _ { 3 } } , } } \\ { { - 1 + \displaystyle \frac { K E \mathbb { 1 } _ { v \geq 0 } + e ^ { \sigma _ { 1 } - d _ { 1 } } } { D _ { 4 } } = - \displaystyle \frac { N - K - 1 + K E \mathbb { 1 } _ { v < 0 } } { D _ { 4 } } , } } \end{array}
$$

where now $D _ { 3 , i } = e ^ { g _ { 1 } - d _ { 1 } } + N - 1 = : D _ { 3 }$ no longer depends on $i ,$ gives the expressions in item 3.

By Theorem C.2, we can aggregate the parameters and deal with a more compact system as follows.

Corollary C.3 (Parameter aggregation). Let $x _ { 1 } = a _ { 2 } , x _ { 2 } = b _ { 1 } , y _ { 1 } = g _ { 1 } - d _ { 1 } , y _ { 2 } = g _ { 2 } - d _ { 2 }$ and $u = x _ { 1 } - x _ { 2 } , v = y _ { 1 } - y _ { 2 }$ . Under the conditions of Theorem C.2, for $t \geq T _ { 0 }$ the quadruple $( x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } )$ solves the closed four-variable system

$$
\dot { x } _ { 1 } = \frac { \alpha } { N } \Big [ \frac { e ^ { y _ { 1 } } + N - 2 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { e ^ { y _ { 1 } } + N - K - 1 + ( i - 1 ) e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { u < 0 } } { D _ { 2 , i } } - \frac { K e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } \Big ]\tag{121}
$$

$$
\dot { x } _ { 2 } = \frac { \alpha } { N ( N - 1 ) } \Big [ \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { u \geq 0 } } { D _ { 2 , i } } + \frac { K e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { u \geq 0 } } { D _ { 4 } } \Big ]\tag{122}
$$

$$
\dot { y } _ { 1 } = \frac { \alpha } { N - 1 } \bigg [ \frac { K ( N - 1 ) } { D _ { 3 } } + \frac { N - K - 1 + K e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { v < 0 } } { D _ { 4 } } - \frac { e ^ { y _ { 1 } } } { D _ { 1 } } - \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { v \geq 0 } + e ^ { y _ { 1 } } } { D _ { 2 , i } } \bigg ]\tag{123}
$$

$$
\dot { y } _ { 2 } = \frac { \alpha } { N - 1 } \bigg [ \sum _ { i = 1 } ^ { K } \frac { ( i - 1 ) e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { v \geq 0 } } { D _ { 2 , i } } + \frac { K e ^ { u _ { + } + v _ { + } } \mathbb { 1 } _ { v \geq 0 } } { D _ { 4 } } \bigg ] ,\tag{124}
$$

where the denominators are

$$
D _ { 1 } = e ^ { x _ { 1 } } + e ^ { y _ { 1 } } + N - 2\tag{125}
$$

$$
D _ { 2 , i } = ( K - i + 1 ) e ^ { x _ { 1 } } + ( i - 1 ) e ^ { u _ { + } + v _ { + } } + e ^ { y _ { 1 } } + N - K - 1\tag{126}
$$

$$
D _ { 3 } = e ^ { y _ { 1 } } + N - 1\tag{127}
$$

$$
D _ { 4 } = K e ^ { u _ { + } + v _ { + } } + e ^ { y _ { 1 } } + N - K - 1 .\tag{128}
$$

The remaining four scalars are recovered afinely: $a _ { 1 } \equiv c _ { 0 } ; b _ { 2 } \equiv 0 f o r t \ge T _ { 0 } ,$ and since $\partial \ell / \partial d _ { i } =$ $- \partial \ell / \partial g _ { i }$ while the learning rates are $\alpha / N$ and $\alpha / ( N ( N - 1 ) )$ , we have $\dot { g } _ { i } = - ( N - 1 ) \dot { d } _ { i }$ , so that $g _ { 1 } + ( N - 1 ) d _ { 1 } \equiv c _ { 0 } + ( N - 1 ) c _ { 0 } ^ { \prime }$ for all $t \geq 0$ and $g _ { 2 } + ( N - 1 ) d _ { 2 }$ is constant for $t \geq T _ { 0 } ,$ hence

$$
g _ { 1 } = \frac { N - 1 } { N } y _ { 1 } + \frac { c _ { 0 } + ( N - 1 ) c _ { 0 } ^ { \prime } } { N } , \qquad d _ { 1 } = - \frac { 1 } { N } y _ { 1 } + \frac { c _ { 0 } + ( N - 1 ) c _ { 0 } ^ { \prime } } { N } ,\tag{129}
$$

$$
g _ { 2 } = \frac { N - 1 } { N } y _ { 2 } + O ( 1 ) , d _ { 2 } = - \frac { 1 } { N } y _ { 2 } + O ( 1 ) ,\tag{130}
$$

where the $O ( 1 )$ terms are constants depending on $N , K , c _ { 0 } , c _ { 0 } ^ { \prime }$ . At time $T _ { 0 }$ the state satisfies $u \geq 0$ $x _ { 2 } > 0 , y _ { 1 } > 0$ and $y _ { 2 } \geq 0$ , and along the system y<sub>2</sub> is non-decreasing.

Proof. By Equation (85), $\begin{array} { r } { \dot { x } _ { 1 } = - \frac { \alpha } { N } \frac { \partial \ell } { \partial a _ { 2 } } , \dot { x } _ { 2 } = - \frac { \alpha } { N \left( N - 1 \right) } \frac { \partial \ell } { \partial b _ { 1 } } } \end{array}$ , and, using $\partial \ell / \partial d _ { 1 } = - \partial \ell / \partial g _ { 1 }$ and, for $i = 2 ,$ , Step 5 of the proof of Theorem C.2,

$$
\dot { y } _ { i } = \dot { g } _ { i } - \dot { d } _ { i } = - \frac { \alpha } { N } \frac { \partial \ell } { \partial g _ { i } } - \frac { \alpha } { N ( N - 1 ) } \frac { \partial \ell } { \partial g _ { i } } = - \frac { \alpha } { N - 1 } \frac { \partial \ell } { \partial g _ { i } } .
$$

Substituting item 3 of Theorem C.2 gives the displayed system; it depends on the eight scalars only through $x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 }$ . The recovery formulas follow from $\dot { g } _ { i } = - ( N - 1 ) \dot { d } _ { i }$ and $g _ { i } - d _ { i } = y _ { i }$ (for $i = 2 , g _ { 2 } + ( N - 1 ) d _ { 2 }$ is constant on $[ T _ { 0 } , \infty )$ by Step 5, and its value difers from the initial value $N c _ { 0 }$ by at most 1, the amount accumulated on $[ 0 , T _ { 0 } ] )$ . The properties of the state at $T _ { 0 }$ are item 1 of Theorem C.2, and $\dot { y } _ { 2 } \geq 0$ is read of the system. □

Before analyzing the system in Theorem C.3, we need the following lemma to characterize the main growth rate. It is applied to Filippov solutions, so the diferential equations are only required to hold almost everywhere, and the coeficient bounds only from some time t<sub>0</sub> on.

Lemma C.4 (Logarithm growth lemma). Let $t _ { 0 } \geq 1$ and let $X , Y : [ t _ { 0 } , \infty )  \mathbb { R }$ be locally absolutely continuous and satisfy, for a.e. $t \geq t _ { 0 }$ 2

$$
X ^ { \prime } = a ( t ) e ^ { Y - X } ,\tag{131}
$$

$$
Y ^ { \prime } = b ( t ) e ^ { - Y } - \widetilde { a } ( t ) e ^ { Y - X } + r ( t ) ,\tag{132}
$$

with measurable coeficients such that, for a.e. $t \geq t _ { 0 }$ 2

$$
0 < a _ { - } \leq a ( t ) \leq a _ { + } < \infty , \qquad 0 < \widetilde { a } _ { - } \leq \widetilde { a } ( t ) \leq \widetilde { a } _ { + } < \infty ,\tag{133}
$$

$$
0 < \beta _ { - } \le \beta ( t ) : = b ( t ) + e ^ { Y ( t ) } r ( t ) \le \beta _ { + } < \infty .\tag{134}
$$

Set $\Theta : = e ^ { 2 Y - X }$ and

$$
\Theta _ { - } : = \frac { 2 \beta _ { - } } { a _ { + } + 2 \widetilde { a } _ { + } } , \qquad \Theta _ { + } : = \frac { 2 \beta _ { + } } { a _ { - } + 2 \widetilde { a } _ { - } } , \qquad c : = \operatorname* { m i n } \{ \Theta ( t _ { 0 } ) , \Theta _ { - } \} , \qquad C : = \operatorname* { m a x } \{ \Theta ( t _ { 0 } ) , \Theta _ { + } \} .
$$

Then for all $t \geq t _ { 0 }$

$$
\begin{array} { r } { c \leq e ^ { 2 Y ( t ) - X ( t ) } \leq C , \qquad e ^ { X ( t _ { 0 } ) / 2 } + \frac { a - \sqrt { c } } { 2 } ( t - t _ { 0 } ) \leq e ^ { X ( t ) / 2 } \leq e ^ { X ( t _ { 0 } ) / 2 } + \frac { a + \sqrt { C } } { 2 } ( t - t _ { 0 } ) . } \end{array}\tag{135}
$$

In particular $X , Y  \infty$ and

$$
X ( t ) = 2 \log t + O ( 1 ) , \quad \quad Y ( t ) = \log t + O ( 1 ) ,\tag{136}
$$

where the $O ( 1 )$ terms depend only on $t _ { 0 } , X ( t _ { 0 } ) , Y ( t _ { 0 } )$ and the coeficient bounds.

Proof. Set $\boldsymbol { P } = e ^ { X }$ and $S = e ^ { Y }$ , so $\Theta = S ^ { 2 } / P > 0$ and, a.e. on $[ t _ { 0 } , \infty )$

$$
{ \cal P } ^ { \prime } = a S , \qquad S ^ { \prime } = \beta - \widetilde { a } { \frac { S ^ { 2 } } { P } } , \qquad ( \log \Theta ) ^ { \prime } = 2 Y ^ { \prime } - X ^ { \prime } = { \frac { 1 } { S } } \bigl \{ 2 \beta - ( a + 2 \widetilde { a } ) \Theta \bigr \} .\tag{137}
$$

Since $S > 0$ , we have $\Theta ^ { \prime } < 0 \mathrm { a . e }$ . on $\{ \Theta > \Theta _ { + } \}$ and $\Theta ^ { \prime } > 0 \mathrm { a . e }$ . on $\left\{ \Theta < \Theta _ { - } \right\}$ : Θ is pushed downward when it is large and upward when it is small. If $\Theta ( t _ { 1 } ) > C$ for some $t _ { 1 } > t _ { 0 }$ , let $t _ { 2 } = \mathrm { s u p } \{ t \in [ t _ { 0 } , t _ { 1 } ]$ $\Theta ( t ) \leq C \}$ ; then $\Theta ( t _ { 2 } ) = C , \Theta > C \geq \Theta _ { + }$ on $( t _ { 2 } , t _ { 1 } ]$ , and $\begin{array} { r } { \Theta ( t _ { 1 } ) = \Theta ( t _ { 2 } ) + \int _ { t _ { 0 } } ^ { t _ { 1 } } \Theta ^ { \prime } \leq C } \end{array}$ , a contradiction. 2 Hence $\Theta \le C$ , and symmetrically $\Theta \ge c > 0$ . Next, $( \sqrt { P } ) ^ { \prime } = P ^ { \prime } / ( 2 \sqrt { P } ) = \textstyle { \frac { 1 } { 2 } } \bar { a } \sqrt { \Theta } \in [ \textstyle { \frac { 1 } { 2 } } a _ { - } \sqrt { c } , \textstyle { \frac { 1 } { 2 } } a _ { + } \sqrt { C } ]$ $\mathrm { { a . e . } }$ , and integrating from $t _ { 0 }$ gives the two-sided bound on $e ^ { X / 2 } = { \sqrt { P } }$ . Finally $X = 2$ log $\sqrt { P }$ and $\begin{array} { r } { Y = \frac { 1 } { 2 } \log \Theta + \log \sqrt { P } } \end{array}$ , and $\begin{array} { r } { \sqrt { P ( t ) } / t \in [ \frac { 1 } { 4 } a _ { - } \sqrt { c } , ~ \sqrt { P ( t _ { 0 } ) } + \frac { 1 } { 2 } a _ { + } \sqrt { C } ] } \end{array}$ for $t \geq 2 t _ { 0 }$ , which proves the claim. □

With the dynamical system in Theorem C.3 and the logarithm growth lemma Theorem C.4, we establish the convergence results in the following theorem.

Theorem C.5. Consider the four-dimensional discontinuous dynamical system of Theorem $C . 3 ,$ with Filippov solutions on the switching surfaces $u = 0$ and $v = 0$ , along the trajectory of Theorem C.2 for $t \geq T _ { 0 } ,$ ; by item 1 of that lemma, $u \geq 0 , x _ { 2 } > 0 , y _ { 1 } > 0$ and $y _ { 2 } \geq 0$ hold along it. Then the following holds, where “bounded” and $O ( 1 )$ refer to constants that do not depend on t but may depend on $N _ { \cdot }$ K and the state at time $T _ { 0 }$

1. The positive part $v _ { + } = ( y _ { 1 } - y _ { 2 } ) _ { + }$ is bounded for $t \geq T _ { 0 }$

2. $x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } \to \infty \ a s \ t \to \infty .$

3. The diferences $2 y _ { 1 } - x _ { 1 }$ and $x _ { 1 } - x _ { 2 }$ are upper bounded for $t \geq T _ { 0 }$

4. $x _ { 1 } , x _ { 2 } = 2 \log ( \alpha t ) + O ( 1 )$ and $y _ { 1 } , y _ { 2 } = \log ( \alpha t ) + O ( 1 )$ , where the $O ( 1 )$ terms do not depend on α.

Proof. Throughout the proof we shift time so that the starting time $T _ { 0 }$ becomes 0; thus $u ( 0 ) \geq 0$ $x _ { 2 } ( 0 ) > 0 , y _ { 1 } ( 0 ) > 0 , y _ { 2 } ( 0 ) \geq 0$ , and, by item 1 of Theorem C.2, $u \geq 0 , x _ { 2 } > 0 , y _ { 1 } > 0 , y _ { 2 } \geq 0$ for all $t \geq 0$ (we nevertheless re-derive $u \geq 0$ below, since only the four-variable system is used from now on). The right-hand side of the system is bounded (every fraction below is at most 1), so Filippov solutions are Lipschitz; all barrier arguments below use only that the relevant quantity is absolutely continuous and that its derivative has a definite sign on an open region, hence they hold for every Filippov solution. We also note that the x-equations do not contain $\mathbb { 1 } _ { v \geq 0 }$ and the y-equations do not contain $\mathbb { 1 } _ { u \geq 0 }$ , so $u ^ { \prime }$ is single-valued on the open set $\{ u < 0 \}$ and $v ^ { \prime }$ is single-valued on $\{ v > 0 \}$ and on $\{ v < 0 \}$ , whatever happens on the other switching surface. Define

$$
G = \frac { e ^ { y _ { 1 } } + N - 2 } { D _ { 1 } } , \qquad H = \frac { e ^ { y _ { 1 } } } { D _ { 1 } } , \qquad E = e ^ { u _ { + } + v _ { + } }\tag{138}
$$

$$
B _ { i } = \frac { e ^ { y _ { 1 } } } { D _ { 2 , i } } , \qquad C _ { i } = \frac { e ^ { y _ { 1 } } + N - K - 1 } { D _ { 2 , i } } , \qquad A _ { i } = \frac { ( i - 1 ) E } { D _ { 2 , i } } ,\tag{139}
$$

$$
Q = \frac { E } { D _ { 4 } } , \qquad R = \frac { N - K - 1 } { D _ { 4 } } , \qquad M = \frac { N - 1 } { D _ { 3 } } .\tag{140}
$$

Also set

$$
\rho = \frac { N } { N - 1 } .\tag{141}
$$

Since multiplying the vector field by the positive constant α only rescales time, we keep α in the formulas but note that the solution with learning rate α at time t coincides with the solution for $\alpha = 1$ at time $\alpha t ;$ hence all constants below are independent of $\alpha$ once the growth is expressed in terms of $\alpha t$

When $u < 0$ , all $\mathbf { 1 } _ { u \geq 0 }$ terms vanish and all $\mathbf { 1 } _ { u < 0 }$ terms are active. Hence

$$
x _ { 2 } ^ { \prime } = 0 ,\tag{142}
$$

whereas

$$
x _ { 1 } ^ { \prime } = \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } ( C _ { i } + A _ { i } ) \right] > 0 .\tag{143}
$$

Therefore $u ^ { \prime } = x _ { 1 } ^ { \prime } - x _ { 2 } ^ { \prime } > 0$ whenever $u < 0$ . Since $u ( 0 ) \geq 0$ , the Filippov forward barrier gives

$$
u ( t ) = x _ { 1 } ( t ) - x _ { 2 } ( t ) \geq 0 .\tag{144}
$$

Indeed, suppose $u ( t _ { 1 } ) < 0$ for some $t _ { 1 } > 0$ and let $t _ { 0 } = \operatorname* { s u p } \{ t < t _ { 1 } : u ( t ) \geq 0 \}$ . Then $u ( t _ { 0 } ) = 0$ and $u < 0$ on $( t _ { 0 } , t _ { 1 } ]$ ; since $x _ { 1 } ^ { \prime } , x _ { 2 } ^ { \prime }$ do not involve $\mathbb { 1 } _ { v \geq 0 }$ , on the open set $\{ u < 0 \}$ every Filippov solution satisfies $u ^ { \prime } = x _ { 1 } ^ { \prime } > 0 \mathrm { a . e . }$ , so $u ( t _ { 1 } ) > u ( t _ { 0 } ) = 0$ , a contradiction. Consequently,

$$
u _ { + } = u , \quad \forall t \geq 0 .\tag{145}
$$

The same one-sided barrier argument is used repeatedly below without further comment.

We first prove that $v _ { + }$ is upper-bounded.

Upper bound of $v _ { + }$ . In order to prove

$$
v _ { + } ( t ) = ( y _ { 1 } ( t ) - y _ { 2 } ( t ) ) _ { + } \leq C .\tag{146}
$$

It is enough to consider the open region $v > 0$ , where $E = e ^ { u + v }$ and

$$
\boldsymbol { v } ^ { \prime } = \frac { \alpha } { N - 1 } \left[ K M + R - H - \sum _ { i = 1 } ^ { K } B _ { i } - 2 \sum _ { i = 1 } ^ { K } A _ { i } - K Q \right] \le \frac { \alpha } { N - 1 } ( K M + R - K Q ) .\tag{147}
$$

To get a negative gradient $v ^ { \prime } ,$ , we split and discuss two cases by fixing a constant $L \geq 1$

Case $1 \colon e ^ { y _ { 2 } - u } \leq L$

Since $y _ { 1 } = y _ { 2 } + v$ and $E = e ^ { u + \tau }$ ，

$$
K Q = { \frac { K e ^ { u + v } } { K e ^ { u + v } + e ^ { y _ { 1 } } + N - K - 1 } } = { \frac { K } { K + e ^ { y _ { 2 } - u } + ( N - K - 1 ) e ^ { - u - v } } } ,\tag{148}
$$

so $K Q$ has a positive lower bound, i.e., $K Q \ge c _ { L } : = K / ( L + N - 1 ) > 0$ (here we used $u + v \geq 0 )$ Meanwhile, for some constant $C = C _ { N , K }$ , we have

$$
K M + R \leq C e ^ { - y _ { 1 } } + C e ^ { - u - v } = C e ^ { - y _ { 2 } - v } + C e ^ { - u - v } \leq 2 C e ^ { - v } ,\tag{149}
$$

where the last step uses $y _ { 2 } \geq 0$ and $u \geq 0$ . Thus $v ^ { \prime } < 0$ once v is larger than a constant depending on L.

Case 2: $e ^ { y _ { 2 } - u } > L$

Put

$$
S : = e ^ { y _ { 2 } - u } .\tag{150}
$$

Then we have $S > L$ . Since $v > 0$ and $u \geq 0$ , we have

$$
E = e ^ { u + v } \geq e ^ { v } , \qquad e ^ { y _ { 1 } } = e ^ { y _ { 2 } + v } = S E .\tag{151}
$$

Consequently

$$
D _ { 4 } = K E + e ^ { y _ { 1 } } + N - K - 1 = E \left( K + S + ( N - K - 1 ) E ^ { - 1 } \right) .\tag{152}
$$

The negative term coming from $D _ { 4 }$ therefore satisfies, once $L \geq 1$

$$
K Q = \frac { { K E } } { { D _ { 4 } } } = \frac { K } { { K + S + ( N - K - 1 ) E ^ { - 1 } } } \ge \frac { K } { ( 2 N ) S } = : \frac { c _ { Q } } { S } ,\tag{153}
$$

where we used $S > L \ge 1 , K \le N - 1$ , and $E ^ { - 1 } \leq 1$ , so $K + S + ( N - K - 1 ) E ^ { - 1 } \leq 2 N S$

On the other hand, the two positive terms in the upper bound for $v ^ { \prime }$ are much smaller:

$$
K M = K \frac { N - 1 } { e ^ { y _ { 1 } } + N - 1 } \leq K \frac { N - 1 } { S E } ,\tag{154}
$$

and, since $D _ { 4 } \geq e ^ { y _ { 1 } } = S E$

$$
R = \frac { N - K - 1 } { D _ { 4 } } \leq \frac { N - K - 1 } { S E } .\tag{155}
$$

Thus

$$
K M + R \leq \frac { C _ { N , K } } { S E } \leq \frac { C _ { N , K } e ^ { - v } } { S } .\tag{156}
$$

Choose $v _ { 1 } = v _ { 1 } ( N , K )$ so large that

$$
C _ { N , K } e ^ { - v _ { 1 } } \leq \frac { c _ { Q } } { 2 } .\tag{157}
$$

Then for every $v \geq v _ { 1 }$ and every point in the case $S > L$ ，

$$
K M + R \leq \frac { c _ { Q } } { 2 S } \leq \frac { 1 } { 2 } K Q .\tag{158}
$$

Returning to the diferential inequality, we have

$$
v ^ { \prime } \leq \frac { \alpha } { N - 1 } ( K M + R - K Q ) \leq - \frac { \alpha } { 2 ( N - 1 ) } K Q < 0 .\tag{159}
$$

This proves that in the whole region $e ^ { y _ { 2 } - u } > L$ , the positive $y \mathrm { - g a p }$ is pushed strictly downward whenever $v \geq v _ { 1 }$

Combining the two cases, there exists $C _ { v }$ such that

$$
v > C _ { v } \quad \Longrightarrow \quad v ^ { \prime } < 0 .\tag{160}
$$

Since $\{ v > C _ { v } \}$ lies strictly inside the open region $\{ v > 0 \}$ , where $v ^ { \prime }$ is single-valued, the one-sided barrier argument gives

$$
v _ { + } = ( y _ { 1 } - y _ { 2 } ) _ { + } \leq C = \operatorname* { m a x } \{ v ( 0 ) , C _ { v } \} , \quad \forall t \geq 0 .\tag{161}
$$

Prove $x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } \to \infty { \bf a s } t \to \infty$ . First observe that, for every Filippov selection (the $\mathbb { 1 } _ { u \geq 0 } .$ -terms cancel in the combination $x _ { 1 } + ( N - 1 ) x _ { 2 } )$

$$
\left( x _ { 1 } + ( N - 1 ) x _ { 2 } \right) ^ { \prime } = \frac { \alpha } { N } \Bigl [ G + \sum _ { i = 1 } ^ { K } ( C _ { i } + A _ { i } ) \Bigr ] \geq \frac { \alpha } { N } G\tag{162}
$$

$$
\ge \frac { \alpha } { N ( N - 2 ) } \frac { e ^ { y _ { 1 } } } { e ^ { x _ { 1 } } + e ^ { y _ { 1 } } + 1 } \ge \frac { \alpha } { N ( N - 2 ) } \frac { 1 } { e ^ { x _ { 1 } + ( N - 1 ) x _ { 2 } } + 2 } ,\tag{163}
$$

where we used $e ^ { x _ { 1 } } + e ^ { y _ { 1 } } + N - 2 \leq ( N - 2 ) ( e ^ { x _ { 1 } } + e ^ { y _ { 1 } } + 1 ) , e ^ { y _ { 1 } } \geq 1$ and $x _ { 2 } \geq 0$ . Writing $w = x _ { 1 } + ( N - 1 ) x _ { 2 }$ this gives $( e ^ { w } + 2 w ) ^ { \prime } \geq c$ , hence w $\geq \log ( c t ) - C$ for large $t , \mathrm { i . e . , } x _ { 1 } + ( N - 1 ) x _ { 2 } \gtrsim \log ( t )$ . Since $x _ { 1 } \geq x _ { 2 }$ , we conclude that $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } x _ { 1 } = + \infty } \end{array}$ . Therefore lim $\begin{array} { r } { \mathfrak { l } _ { t  \infty } D _ { 1 } = + \infty , \operatorname* { l i m } _ { t  \infty } D _ { 2 , i } = + \infty . } \end{array}$ , and consequently $H  0$ and $B _ { i } \to 0$ as long as $y _ { 1 }$ stays bounded.

Next we show that $y _ { 1 }$ is unbounded. Note that $y _ { 1 } ^ { \prime }$ contains the negative term $- \mathbb { 1 } _ { v \geq 0 } \sum _ { i } A _ { i }$ which need not be small at this stage, because $A _ { i } \to 0$ only once $x _ { 2 } \to \infty ;$ we therefore consider $y _ { 1 } + y _ { 2 }$ , whose derivative does not depend on $\mathbb { 1 } _ { v \geq 0 }$ and hence is the same for every Filippov selection on $v = 0 \mathrm { : }$

$$
( y _ { 1 } + y _ { 2 } ) ^ { \prime } = \frac { \alpha } { N - 1 } \Big [ K M + R - H - \sum _ { i = 1 } ^ { K } B _ { i } + K Q \Big ] \ \geq \ \frac { \alpha } { N - 1 } \Big [ K M - H - \sum _ { i = 1 } ^ { K } B _ { i } \Big ] .\tag{164}
$$

Suppose $y _ { 1 } \le B$ for all t. Since $y _ { 2 } ^ { \prime } = 0$ on $\{ v \ : < \ : 0 \}$ , y<sub>2</sub> can increase only while $y _ { 2 } \leq y _ { 1 } \leq B ;$ hence $y _ { 2 } ( t ) \leq \operatorname* { m a x } \{ y _ { 2 } ( 0 ) , B \}$ for all $t \ ( \mathrm { i f } \ y _ { 2 } ( t _ { 1 } ) > \operatorname* { m a x } \{ y _ { 2 } ( 0 ) , B \}$ , let $t _ { 0 } < t _ { 1 }$ be the last time with $y _ { 2 } \leq \operatorname* { m a x } \{ y _ { 2 } ( 0 ) , B \}$ ; on $( t _ { 0 } , t _ { 1 } ]$ we have $y _ { 2 } > B \geq y _ { 1 } , \mathrm { i . e . ~ } v < 0$ , so $y _ { 2 } ^ { \prime } = 0 ~ \mathrm { a . e }$ . there, a contradiction). Thus $y _ { 1 } + y _ { 2 }$ is bounded. On the other hand $M \ge ( N - 1 ) / ( e ^ { B } + N - 1 ) = : c _ { B } > 0$ depends only on $B .$ while $\begin{array} { r } { H + \sum _ { i } B _ { i } \le e ^ { B } \big ( 1 / D _ { 1 } + \sum _ { i } 1 / D _ { 2 , i } \big ) \to 0 } \end{array}$ because x<sub>1</sub> $ \infty$ . Hence $\begin{array} { r } { ( y _ { 1 } + y _ { 2 } ) ^ { \prime } \ge \frac { \alpha } { N - 1 } \frac { K c _ { B } } { 2 } > 0 } \end{array}$ for all large $t ,$ so $y _ { 1 } + y _ { 2 } \to \infty$ , a contradiction. Therefore lim su $\mathrm { p } _ { t } y _ { 1 } = + \infty$ . Since $y _ { 2 } \geq y _ { 1 } - v _ { + } \geq y _ { 1 } - C$ also lim $\operatorname* { s u p } _ { t } y _ { 2 } = + \infty$ , and since $y _ { 2 }$ is non-decreasing, lim $y _ { 2 } = + \infty$

For $x _ { 2 } ,$ suppose it is bounded. Since $x _ { 1 }  \infty , u > 0$ for all large $t ,$ , and on $\{ u > 0 \}$ we have $\begin{array} { r } { x _ { 2 } ^ { \prime } = \frac { \alpha } { N ( N - 1 ) } [ \sum _ { i } A _ { i } + K Q ] } \end{array}$ while $\begin{array} { r } { y _ { 2 } ^ { \prime } = \eta \frac { \alpha } { N - 1 } \bigl [ \sum _ { i } A _ { i } + K Q \bigr ] \leq N x _ { 2 } ^ { \prime } } \end{array}$ for every Filippov weight $\eta \in [ 0 , 1 ]$ of $\mathbb { 1 } _ { v \geq 0 }$ . Integrating, $x _ { 2 } ( t ) \geq x _ { 2 } ( T ) + ( y _ { 2 } ( t ) - y _ { 2 } ( T ) ) / N \to + \infty$ , which contradicts the boundedness assumption. Therefore, lim<sub>t</sub> $x _ { 2 } ( t ) \to + \infty$

For $y _ { 1 } , \forall B > 0$ , we prove that there exists a time $t _ { 0 }$ such that $y _ { 1 } \geq B$ for all $t \geq t _ { 0 }$ . On the region $\{ v < 0 , \ y _ { 1 } \leq B \}$ there is no $A _ { i } { \mathrm { - t e r m } }$ in $y _ { 1 } ^ { \prime }$ , and

$$
y _ { 1 } ^ { \prime } = { \frac { \alpha } { N - 1 } } { \Big [ } K M + R - H - \sum _ { i } B _ { i } + K Q { \Big ] } \geq { \frac { \alpha } { N - 1 } } { \Big [ } K c _ { B } - e ^ { B } { \Big ( } { \frac { 1 } { D _ { 1 } } } + \sum _ { i } { \frac { 1 } { D _ { 2 , i } } } { \Big ) } { \Big ] } \geq c > 0\tag{165}
$$

for all $t \geq t _ { 1 }$ , where $t _ { 1 }$ is chosen, using $x _ { 1 } \to \infty$ , so that the bracket is at least $K c _ { B } / 2$ . Since lim<sub>t</sub> $y _ { 2 } =$ $+ \infty$ , pick $t _ { 2 } \geq t _ { 1 }$ with $y _ { 2 } ( t _ { 2 } ) > B$ . For $t \geq t _ { 2 }$ , whenever $y _ { 1 } ( t ) \le B$ we have $v ( t ) \leq B - y _ { 2 } ( t _ { 2 } ) < 0$ because y is non-decreasing, so the state lies in $\{ v < 0 , y _ { 1 } \le B \}$ and $y _ { 1 } ^ { \prime } \ge c > 0$ there. Hence $y _ { 1 } \geq B$ for all $t \geq t _ { 0 } : = t _ { 2 } + ( B - y _ { 1 } ( t _ { 2 } ) ) _ { + } / c$ , by the barrier argument. Thus lim<sub>t</sub> $y _ { 1 } ( t ) \to + \infty$

Upper bound of $2 y _ { 1 } - x _ { 1 }$ . To analyze the behavior of $2 y _ { 1 } - x _ { 1 }$ , we define new variables

$$
s = e ^ { u } = e ^ { x _ { 1 } - x _ { 2 } } , \qquad Z = e ^ { 2 y _ { 1 } - x _ { 1 } } .\tag{166}
$$

The desired gap estimates are

$$
s \leq C , \quad \quad Z \leq C .\tag{167}
$$

Indeed, $s \leq C$ is equivalent to $x _ { 1 } - x _ { 2 } \leq C$ , and $Z \leq C$ is equivalent to $2 y _ { 1 } - x _ { 1 } \leq C .$

To prove the upper bound, we use a Lyapunov argument. Define the Lyapunov product

$$
W = Z s ^ { p } .\tag{168}
$$

Choose

$$
3 - \frac { 1 } { N } < p < 1 + \frac { 2 N } { N - 1 } .\tag{169}
$$

The interval is nonempty because

$$
1 + { \frac { 2 N } { N - 1 } } - \left( 3 - { \frac { 1 } { N } } \right) = { \frac { 2 } { N - 1 } } + { \frac { 1 } { N } } = { \frac { 3 N - 1 } { N ( N - 1 ) } } > 0 .\tag{170}
$$

We discuss the regions where $u > 0$ or $u = 0$ separately.

Region 1: $u > 0$ . We have $\mathbf { 1 } _ { u \geq 0 } = 1$ and $\mathbf { 1 } _ { u < 0 } = 0$ . Thus

$$
x _ { 1 } ^ { \prime } = \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } C _ { i } - K Q \right] ,\tag{171}
$$

$$
x _ { 2 } ^ { \prime } = \frac { \alpha } { N ( N - 1 ) } \left[ \sum _ { i = 1 } ^ { K } A _ { i } + K Q \right] .\tag{172}
$$

Therefore

$$
\frac { s ^ { \prime } } { s } = u ^ { \prime } = \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } C _ { i } - \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { K } A _ { i } - \frac { K N } { N - 1 } Q \right] .\tag{173}
$$

Since v may not be positive and can hit the sliding boundary, we introduce the Filippov parameter $0 \leq \eta \leq 1$ , where η is the convex value of $\mathbf { 1 } _ { v > 0 }$ . Then every admissible velocity on $u > 0$ satisfies

$$
y _ { 1 } ^ { \prime } = \frac { \alpha } { N - 1 } \left[ K M + R - H - \sum _ { i = 1 } ^ { K } B _ { i } - \eta \sum _ { i = 1 } ^ { K } A _ { i } + ( 1 - \eta ) K Q \right] .\tag{174}
$$

Consequently

$$
\begin{array} { l } { \displaystyle { \frac { Z ^ { \prime } } { Z } = 2 y _ { 1 } ^ { \prime } - x _ { 1 } ^ { \prime } } } \\ { \displaystyle { \quad = \frac { \alpha } { N } \Bigg [ 2 \rho K M + 2 \rho R - 2 \rho H - G - \sum _ { i = 1 } ^ { K } \{ 2 \rho B _ { i } + C _ { i } \} - 2 \rho \eta \sum _ { i = 1 } ^ { K } A _ { i } + K \{ 1 + 2 \rho ( 1 - \eta ) \} Q \Bigg ] , } } \end{array}\tag{175}
$$

where $\rho = N / ( N - 1 )$ . Therefore

$$
\begin{array} { c }  { \displaystyle { \frac { W ^ { \prime } } { W } = \frac { Z ^ { \prime } } { Z } + p \frac { s ^ { \prime } } { s } = \frac { \alpha } { N } \Bigg [ 2 \rho K M + 2 \rho R + \left( ( p - 1 ) G - 2 \rho H \right) + \sum _ { i = 1 } ^ { K } \left( p C _ { i } - C _ { i } - 2 \rho B _ { i } \right) } } \\ { { - \left( 2 \rho \eta + \frac { p } { N - 1 } \right) \sum _ { i = 1 } ^ { K } A _ { i } + K \left( 1 + 2 \rho ( 1 - \eta ) - \frac { p N } { N - 1 } \right) Q \Bigg ] . } } \end{array}\tag{176}
$$

The coeficient of $Q$ is largest when $\eta = 0$ , and the lower bound $p > 3 - 1 / N$ is exactly

$$
1 + 2 \rho - \frac { p N } { N - 1 } < 0 .\tag{177}
$$

The coeficient of $\textstyle \sum _ { i } A _ { i }$ is strictly negative. The indexed $D _ { 2 , i }$ contribution can be written as

$$
p C _ { i } - C _ { i } - 2 \rho B _ { i } = \frac { ( p - 1 ) ( N - K - 1 ) + ( p - 1 - 2 \rho ) e ^ { y _ { 1 } } } { D _ { 2 , i } } .\tag{178}
$$

Since $p < 1 + 2 \rho$ , the coeficient of $e ^ { y _ { 1 } }$ is strictly negative. The same calculation for $D _ { 1 }$ gives

$$
( p - 1 ) G - 2 \rho H = \frac { ( p - 1 ) ( N - 2 ) + ( p - 1 - 2 \rho ) e ^ { y _ { 1 } } } { D _ { 1 } } ,\tag{179}
$$

which has the same structure.

We now estimate each term in Equation (176).

Define

$$
a = e ^ { x _ { 1 } } , \qquad b = e ^ { y _ { 1 } } , \qquad s = e ^ { u } .\tag{180}
$$

Since $v _ { + }$ is bounded, there are constants $0 < c _ { E } < C _ { E } < \infty$ , depending only on the previously obtained $v _ { + }$ bound, such that

$$
c _ { E } s \leq E = s e ^ { v _ { + } } \leq C _ { E } s .\tag{181}
$$

Put

$$
\delta _ { y } : = 1 + 2 \rho - p > 0 , \qquad \delta _ { Q } : = p \rho - ( 1 + 2 \rho ) > 0 , \qquad c _ { A } : = \frac { p } { N - 1 } > 0 .\tag{182}
$$

The two strict inequalities follow exactly from the choice $3 - 1 / N < p < 1 + 2 N / ( N - 1 )$

Starting from the exact formula for $W ^ { \prime } / W$ , we separate the terms with a fixed negative coeficient from the terms with a constant numerator. For the $D _ { 1 ^ { - } } \mathrm { t e r m }$ 2

$$
( p - 1 ) G - 2 \rho H = \frac { ( p - 1 ) ( N - 2 ) - \delta _ { y } b } { D _ { 1 } } \leq - \delta _ { y } H + C _ { N , K , p } \frac { 1 } { D _ { 1 } } .\tag{183}
$$

For every $i ,$

$$
p C _ { i } - C _ { i } - 2 \rho B _ { i } = \frac { ( p - 1 ) ( N - K - 1 ) - \delta _ { y } b } { D _ { 2 , i } } \leq - \delta _ { y } B _ { i } + C _ { N , K , p } \frac { 1 } { D _ { 2 , i } } .\tag{184}
$$

Moreover the coeficient of $\textstyle \sum _ { i } A _ { i }$ is at most $- c A$ , and the coeficient of $K Q$ is maximal at $\eta = 0$ where it equals $- K \delta _ { Q }$ . Therefore, after enlarging constants, every Filippov velocity in the region $u > 0$ satisfies

$$
\begin{array} { r c l } { \displaystyle \frac { N } { \alpha } \frac { W ^ { \prime } } { W } \le \mathcal { P } - \mathcal { N } , } \\ { \displaystyle \mathcal { P } : = C \left( M + R + \frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } \right) , } \\ { \displaystyle \mathcal { N } : = c \left( H + \sum _ { i = 1 } ^ { K } B _ { i } + \sum _ { i = 1 } ^ { K } A _ { i } + Q \right) , } \end{array}\tag{185}
$$

where $\mathcal { P }$ collects the terms with constant numerators and $\mathcal { N }$ the terms with a fixed negative coeficient (the letter N is reserved for the number of vertices). Thus it is enough to show that $\mathcal { N }$ dominates P whenever

$$
W = Z s ^ { p } = { \frac { b ^ { 2 } s ^ { p } } { a } }\tag{186}
$$

is large.

We record the elementary estimates used below; recall $a = e ^ { x _ { 1 } } \geq 1 , b = e ^ { y _ { 1 } } \geq 1 , s = e ^ { u } \geq 1$ and $E \asymp s$ . Since $K - i + 1 \geq 1$ in every $D _ { 2 , i }$ and $N - K - 1 \geq 4$

$$
M \leq \frac { N - 1 } { b + 1 } , \qquad R \leq \frac { N - K - 1 } { b + 1 } , \qquad \frac { 1 } { D _ { 1 } } + \sum _ { i } \frac { 1 } { D _ { 2 , i } } \leq \frac { K + 1 } { b + 1 } , \qquad \mathrm { h e n c e } \qquad \mathcal { P } \leq \frac { C } { b + 1 } .\tag{187}
$$

For the negative part,

$$
Q = \frac { E } { K E + b + N - K - 1 } \asymp \frac { s } { s + b + 1 } ,\tag{188}
$$

$$
H = \frac { b } { a + b + N - 2 } \asymp \frac { b } { a + b + 1 } , \qquad B _ { 1 } = \frac { b } { K a + b + N - K - 1 } \asymp \frac { b } { a + b + 1 } ,\tag{189}
$$

so that, dropping the nonnegative terms $A _ { i }$ and $B _ { i } , i \geq 2$

$$
\mathcal { N } \geq c \left( \frac { b } { a + b + 1 } + \frac { s } { s + b + 1 } \right) .\tag{190}
$$

We now show that for $b \geq 1 , s \geq 1$ and $W = b ^ { 2 } s ^ { p } / a$

$$
\frac { b } { a + b + 1 } + \frac { s } { s + b + 1 } \ \geq \ \frac { 1 } { 4 } \operatorname* { m i n } \left\{ 1 , \frac { W ^ { 1 / ( p + 1 ) } } { b + 1 } \right\} .\tag{191}
$$

Indeed, if $s \geq b + 1$ the second term is at least $\frac { 1 } { 2 } ; \mathrm { i f } a \le b + 1$ the first term is at least $\textstyle b / ( 2 ( b + 1 ) ) \geq { \frac { 1 } { 4 } }$ Otherwise $a > b + 1 \geq s .$ , so $b / ( a + b + 1 ) \geq \bar { b } / ( 3 a ) = W / ( 3 b s ^ { p } )$ and $s / ( s + b + 1 ) \geq s / ( 3 ( b + 1 ) { \overset { } { ) } }$ and by the weighted AM–GM inequality, for all $s > 0 .$

$$
\frac { W } { b s ^ { p } } + \frac { s } { b + 1 } \geq \Big ( \frac { W } { b } \Big ) ^ { \frac { 1 } { p + 1 } } ( b + 1 ) ^ { - \frac { p } { p + 1 } } \geq \frac { W ^ { 1 / ( p + 1 ) } } { b + 1 } ,\tag{192}
$$

where we used $b \leq b + 1$ . This proves Equation (191).

Combining Equations (185), (187), (190) and (191), every Filippov velocity in the region $u > 0$ satisfies

$$
\frac { N } { \alpha } \frac { W ^ { \prime } } { W } \ \leq \ \frac { C } { b + 1 } - c ^ { \prime } \operatorname* { m i n } \left\{ 1 , \frac { W ^ { 1 / ( p + 1 ) } } { b + 1 } \right\} .\tag{193}
$$

Since lim<sub>t</sub> $y _ { 1 } = + \infty$ , we can pick a time $t _ { 0 }$ such that $b = e ^ { y _ { 1 } } \geq B _ { 0 } : = 2 C / c ^ { \prime }$ for all $t \geq t _ { 0 }$ . Then, for $t \geq t _ { 0 }$ , the right-hand side of Equation (193) is negative whenever $W \ge C _ { W } : = ( 2 C / c ^ { \prime } ) ^ { p + 1 }$ : if the minimum equals 1 this follows from $b \geq B _ { 0 }$ , and otherwise $c ^ { \prime } W ^ { 1 / ( p + 1 ) } / ( b + 1 ) \geq 2 C / ( b + 1 )$

We conclude that for $t \geq t _ { 0 }$ , in the region $u > 0 , W ^ { \prime } < 0$ whenever $W \geq C _ { W } ;$ in particular a trajectory cannot cross the level surface $W = C _ { W }$ outward while $u > 0$

Region $u = 0$ . We now analyze the $u = 0$ region. The gradient is

$$
\frac { \mathrm { d } } { \mathrm { d } t } \log W = 2 y _ { 1 } ^ { \prime } - x _ { 1 } ^ { \prime } + p ( x _ { 1 } ^ { \prime } - x _ { 2 } ^ { \prime } ) = 2 y _ { 1 } ^ { \prime } + ( p - 1 ) x _ { 1 } ^ { \prime } - p x _ { 2 } ^ { \prime }\tag{194}
$$

for every admissible Filippov boundary velocity.

At $u = 0$ we have

$$
s = 1 , \qquad W = Z = \frac { e ^ { 2 y _ { 1 } } } { e ^ { x _ { 1 } } } , \qquad E = e ^ { v _ { + } } .\tag{195}
$$

Since $v _ { + }$ is bounded, there are constants $0 < c _ { E } < C _ { E } < \infty$ such that

$$
c _ { E } \leq E \leq C _ { E }\tag{196}
$$

on the whole boundary $u = 0$ . Introduce the Filippov parameter $0 \leq \theta \leq 1$ for the u-discontinuity, so that θ is the convex value of $\mathbf { 1 } _ { u \geq 0 }$ and $1 - \theta$ is the convex value of $\mathbf { 1 } _ { u < 0 }$ . As before, $0 \leq \eta \leq 1$ denotes the convex value of $\mathbf { 1 } _ { v \geq 0 }$ . Then every admissible boundary velocity satisfies

$$
\begin{array} { l } { { \displaystyle x _ { 1 } ^ { \prime } = \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } C _ { i } + ( 1 - \theta ) \sum _ { i = 1 } ^ { K } A _ { i } - \theta K Q \right] , } } \\ { { \displaystyle x _ { 2 } ^ { \prime } = \frac { \alpha \theta } { N ( N - 1 ) } \left[ \sum _ { i = 1 } ^ { K } A _ { i } + K Q \right] , } } \\ { { \displaystyle y _ { 1 } ^ { \prime } = \frac { \alpha } { N - 1 } \left[ K M + R - H - \sum _ { i = 1 } ^ { K } B _ { i } - \eta \sum _ { i = 1 } ^ { K } A _ { i } + ( 1 - \eta ) K Q \right] . } } \end{array}\tag{197}
$$

Therefore, with $\rho = N / ( N - 1 )$ ),

$$
\begin{array} { r l } { \displaystyle { \frac { N } { \alpha } \frac { \mathrm { d } } { \mathrm { d } t } \log { W } = 2 \rho ( K M + R ) + \left( ( p - 1 ) G - 2 \rho H \right) + \sum _ { i = 1 } ^ { K } \left( ( p - 1 ) C _ { i } - 2 \rho B _ { i } \right) } } & { { } } \\ { \displaystyle { + \left( ( p - 1 ) ( 1 - \theta ) - 2 \rho \eta - \frac { p \theta } { N - 1 } \right) \sum _ { i = 1 } ^ { K } A _ { i } } } & { { } } \\ { \displaystyle { + K \left( 2 \rho ( 1 - \eta ) - ( p - 1 ) \theta - \frac { p \theta } { N - 1 } \right) Q . } } & { { } } \end{array}\tag{198}
$$

The two denominator terms involving $D _ { 1 }$ and $D _ { 2 , i }$ have the same favorable structure as in the interior calculation. Namely, with

$$
a = e ^ { x _ { 1 } } , \qquad b = e ^ { y _ { 1 } } , \qquad \delta _ { y } = 1 + 2 \rho - p > 0 ,\tag{199}
$$

we have

$$
( p - 1 ) G - 2 \rho H = \frac { ( p - 1 ) ( N - 2 ) - \delta _ { y } b } { D _ { 1 } } \leq - \delta _ { y } H + C \frac { 1 } { D _ { 1 } } ,\tag{200}
$$

and, for every i,

$$
( p - 1 ) C _ { i } - 2 \rho B _ { i } = \frac { ( p - 1 ) ( N - K - 1 ) - \delta _ { y } b } { D _ { 2 , i } } \leq - \delta _ { y } B _ { i } + C \frac { 1 } { D _ { 2 , i } } .\tag{201}
$$

The coeficients of the $A _ { i }$ and $Q$ terms in Equation (198) are uniformly bounded above, independently of θ and η. Hence Equations (198), (200) and (201) imply

$$
\frac { N } { \alpha } \frac { \mathrm { d } } { \mathrm { d } t } \log { W } \leq C \left( M + R + \frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } + \sum _ { i = 1 } ^ { K } A _ { i } + Q \right) - c \left( H + \sum _ { i = 1 } ^ { K } B _ { i } \right) .\tag{202}
$$

This estimate is valid for every Filippov choice of θ and $\eta$ on $u = 0$

We now show that the negative term in Equation (202) dominates whenever $W = Z$ is large. Since $y _ { 1 } ( t ) \to \infty$ , choose $t _ { 0 }$ so large that

$$
b = e ^ { y _ { 1 } ( t ) } \geq B _ { 0 }\tag{203}
$$

for all $t \geq t _ { 0 }$ , where $B _ { 0 }$ will be fixed below. On $u = 0$ and for $t \geq t _ { 0 }$ , E is bounded and b is large, so

$$
M \leq \frac { C } { b } , \qquad R \leq \frac { C } { b } , \qquad Q \leq \frac { C } { b } .\tag{204}
$$

Moreover, since every $D _ { 2 , i }$ contains the term $( K - i + 1 )$ a and $K - i + 1 \geq 1$ , while E is bounded,

$$
\sum _ { i = 1 } ^ { K } A _ { i } \leq \frac { C } { a + b + 1 } ,\tag{205}
$$

and

$$
\frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } \leq \frac { C } { a + b + 1 } .\tag{206}
$$

Finally,

$$
H + \sum _ { i = 1 } ^ { K } B _ { i } \geq c { \frac { b } { a + b + 1 } } .\tag{207}
$$

Combining Equation (202)–Equation (207), every admissible boundary velocity satisfies

$$
\frac { N } { \alpha } \frac { \mathrm { d } } { \mathrm { d } t } \log { W } \leq \frac { C } { b } + \frac { C } { a + b + 1 } - c \frac { b } { a + b + 1 } .\tag{208}
$$

It remains only to check the two possible relative sizes of a and b.

Boundary case 1: $a \leq L ( b + 1 )$ . Choose $L > 1$ fixed. Then

$$
{ \frac { b } { a + b + 1 } } \geq c _ { L } > 0 .\tag{209}
$$

Thus the negative term in Equation (208) is bounded below by a positive constant, whereas the positive terms are $O ( b ^ { - 1 } )$ . After increasing $B _ { 0 }$ if necessary, Equation (208) gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \log W < 0\tag{210}
$$

throughout this boundary subcase.

Boundary case 2: $a > L ( b + 1 )$ . Here

$$
{ \frac { b } { a + b + 1 } } \asymp { \frac { b } { a } } .\tag{211}
$$

Since $W = Z = b ^ { 2 } / a$ on $u = 0$ , this becomes

$$
{ \frac { b } { a } } = { \frac { W } { b } } .\tag{212}
$$

Also $( a + b + 1 ) ^ { - 1 } \leq C / a \leq C / b$ . Hence Equation (208) yields

$$
\frac { N } { \alpha } \frac { \mathrm { d } } { \mathrm { d } t } \log W \leq \frac { C } { b } - c \frac { W } { b } .\tag{213}
$$

Therefore the boundary derivative is strictly negative in this subcase whenever $W \geq C _ { \partial }$ , where $C _ { \partial }$ depends only on the fixed parameters and on the bound for $v _ { + }$

The two boundary cases show that, for $t \geq t _ { 0 }$ , no Filippov velocity at $u = 0$ can point outward through a suficiently large level surface of W. More precisely, after choosing $C _ { \partial }$ and $B _ { 0 }$ large enough,

$$
W \geq C _ { \partial } \quad \Longrightarrow \quad { \frac { \mathrm { d } } { \mathrm { d } t } } \log W < 0\tag{214}
$$

for every admissible boundary velocity. This is the required boundary version of the Lyapunov estimate.

Combining the interior and boundary estimates: for a.e. $t \geq t _ { 0 } , W ( t ) \geq C _ { * } : = \operatorname* { m a x } \{ C _ { W } , C _ { \partial } \}$ implies (log $W ) ^ { \prime } ( t ) < 0$ , whether $u ( t ) > 0 \ \mathrm { o r } \ u ( t ) = 0$ . Since $W$ is absolutely continuous (the vector field is bounded, so $x _ { i } , y _ { i }$ are Lipschitz), the one-sided barrier argument gives $W ( t ) \leq \operatorname* { m a x } \{ W ( t _ { 0 } ) , C _ { * } \}$ for all $t \geq t _ { 0 }$ , and W is bounded on the compact interval $[ 0 , t _ { 0 } ]$ by continuity. Hence

$$
W ( t ) = Z ( t ) s ( t ) ^ { p } \leq C \qquad \forall t \geq 0 .\tag{215}
$$

Since $s \geq 1$ , we immediately obtain

$$
Z ( t ) \leq C .\tag{216}
$$

Equivalently,

$$
2 y _ { 1 } ( t ) - x _ { 1 } ( t ) \leq C _ { Z } \qquad \forall t \geq 0 .\tag{217}
$$

Upper bound of $x _ { 1 } - x _ { 2 }$ . We now prove that the upper barrier for

$$
s = e ^ { u } = e ^ { x _ { 1 } - x _ { 2 } }\tag{218}
$$

is finite. We already know $u \geq 0$ , hence $s \geq 1$ . It remains to exclude escape of s to infinity.

It sufices to consider the region $u > 0$ . There $E = e ^ { u + v _ { + } } = s e ^ { v _ { + } }$ , and the formulae for $x _ { 1 } ^ { \prime }$ and $x _ { 2 } ^ { \prime }$ give

$$
\begin{array} { l } { { \displaystyle { u ^ { \prime } = x _ { 1 } ^ { \prime } - x _ { 2 } ^ { \prime } } } } \\ { { \displaystyle { \quad = \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } C _ { i } - \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { K } A _ { i } - \frac { K N } { N - 1 } Q \right] \le \frac { \alpha } { N } \Big [ G + \sum _ { i = 1 } ^ { K } C _ { i } - Q \Big ] , } } } \end{array}\tag{219}
$$

where we dropped $A _ { i } \geq 0$ and used $K N / ( N - 1 ) \geq 1$ . Since $y _ { 1 } \geq 0$ and $x _ { 2 } \geq 0$ , we have $b \geq 1$ and $s \leq a ,$ so for the positive terms $G$ and $C _ { i }$

$$
G \leq C { \frac { b + 1 } { a + b + 1 } } \leq C { \frac { b } { a } } , \qquad C _ { i } \leq C { \frac { b + 1 } { a + b + 1 } } \leq C { \frac { b } { a } } .\tag{220}
$$

For $Q ,$ since $E \le e ^ { C _ { v } } s$ we have the lower bound

$$
Q \geq C ^ { \prime } { \frac { s } { s + b + 1 } } .\tag{221}
$$

Therefore $u ^ { \prime } < 0$ as soon as

$$
( K + 1 ) C _ { a } ^ { \underline { { b } } } < C ^ { \prime } { \frac { s } { s + b + 1 } } , \qquad \mathrm { i . e . } \qquad ( K + 1 ) C \bigl ( 1 + e ^ { y _ { 1 } - u } + e ^ { - u } \bigr ) < C ^ { \prime } e ^ { x _ { 1 } - y _ { 1 } } .\tag{222}
$$

We claim that Equation (222) holds whenever

$$
u \ \ge \ 2 y _ { 1 } - x _ { 1 } + C _ { 2 } , \qquad C _ { 2 } : = 2 \log \bigl ( 3 ( K + 1 ) C / C ^ { \prime } \bigr ) .\tag{223}
$$

Indeed, since $x _ { 2 } \geq 0$ we have $x _ { 1 } \geq u ,$ so Equation (223) implies $x _ { 1 } \geq 2 y _ { 1 } - x _ { 1 } + C _ { 2 } , { \mathrm { i . e . ~ } } x _ { 1 } - y _ { 1 } \geq C _ { 2 } / 2$ hence $1 + e ^ { - u } \le 2 \le 2 e ^ { - C _ { 2 } / 2 } e ^ { x _ { 1 } }$ <sup>−y1</sup> and $e ^ { y _ { 1 } - u } \leq e ^ { - C _ { 2 } } e ^ { x _ { 1 } - y _ { 1 } } \leq e ^ { - C _ { 2 } / 2 } e ^ { x _ { 1 } - y _ { 1 } }$ , so the left-hand side of Equation (222) is at most $3 ( K + 1 ) C e ^ { - C _ { 2 } / 2 } e ^ { x _ { 1 } - y _ { 1 } } = C ^ { \prime } e ^ { x _ { 1 } - y _ { 1 } }$ , with strict inequality after enlarging $C _ { 2 }$ slightly. (Note that the two extra terms $1 + e ^ { - u }$ cannot be dropped: without the input $x _ { 1 } \geq u$ the implication fails.)

Therefore, we argue that

$$
u ( t ) \leq U ( t ) : = \operatorname* { m a x } \Bigl \{ u ( 0 ) , \ \operatorname* { s u p } _ { 0 \leq \tau \leq t } \bigl ( 2 y _ { 1 } ( \tau ) - x _ { 1 } ( \tau ) \bigr ) + C _ { 2 } , \ \epsilon \Bigr \} ,\tag{224}
$$

where $\epsilon$ is a small positive constant. To prove this, suppose the opposite and consider the first violation time

$$
t _ { 0 } = \operatorname* { i n f } \{ t : u ( t ) > U ( t ) \} .\tag{225}
$$

Since u and $U$ are continuous, $u ( t _ { 0 } ) = U ( t _ { 0 } )$ , and $U$ is non-decreasing. Since $u ( t _ { 0 } ) \geq \epsilon > 0$ , the state lies in the open region $\{ u > 0 \}$ , where u is continuously diferentiable (the x-equations contain no v-indicator and $E = e ^ { u + v _ { + } }$ is continuous), and $u ( t _ { 0 } ) \geq 2 y _ { 1 } ( t _ { 0 } ) - x _ { 1 } ( t _ { 0 } ) + C _ { 2 }$ gives $u ^ { \prime } ( t _ { 0 } ) < 0$ by Equations (222) and (223). Hence $u ( t ) < u ( t _ { 0 } ) = U ( t _ { 0 } ) \leq U ( t )$ on a right neighborhood of $t _ { 0 }$ contradicting the definition of $t _ { 0 }$

Further, since $2 y _ { 1 } - x _ { 1 } \le C _ { Z }$ for all $t \geq 0$ , we conclude that u is also upper-bounded, $i . e .$

$$
0 \leq x _ { 1 } - x _ { 2 } \leq C _ { u } : = \operatorname* { m a x } \{ u ( 0 ) , C _ { Z } + C _ { 2 } , \epsilon \} , \quad \forall t \geq 0 .\tag{226}
$$

Logarithm growth rate The averaged x-variable for the system is

$$
X = { \frac { x _ { 1 } + ( N - 1 ) x _ { 2 } } { N } } , \qquad Y = y _ { 1 } .\tag{227}
$$

Since $0 \leq u = x _ { 1 } - x _ { 2 } \leq C _ { u }$

$$
\begin{array} { r } { X = x _ { 1 } + O ( 1 ) = x _ { 2 } + O ( 1 ) , \qquad \mathrm { m o r e ~ p r e c i s e l y } \qquad X = x _ { 1 } - \frac { N - 1 } { N } u . } \end{array}\tag{228}
$$

Also u and $v _ { + }$ are bounded, so

$$
1 \leq E = e ^ { u + v _ { + } } \leq C _ { E } .\tag{229}
$$

The estimate $Z = e ^ { 2 Y - x _ { 1 } } \leq C$ gives

$$
e ^ { 2 Y } \leq C e ^ { x _ { 1 } } \asymp C e ^ { X } ,\tag{230}
$$

and in particular, since $Y  \infty$

$$
e ^ { y _ { 1 } - x _ { 1 } } = Z e ^ { - y _ { 1 } } \le C e ^ { - Y } \to 0 , \qquad e ^ { - X } \le e ^ { \frac { N - 1 } { N } C _ { u } } e ^ { - x _ { 1 } } \le C e ^ { - 2 Y } \le C e ^ { - Y } .\tag{231}
$$

Consequently $D _ { 1 } = e ^ { x _ { 1 } } ( 1 + o ( 1 ) )$ and $D _ { 2 , i } = ( K - i + 1 ) e ^ { x _ { 1 } } ( 1 + o ( 1 ) )$ ; this is where the bound $2 y _ { 1 } - x _ { 1 } \leq C$ enters.

Writing $\theta \in [ 0 , 1 ]$ for the Filippov weight of $\mathbb { 1 } _ { u > 0 }$ (so $\theta = 1$ on $\{ u > 0 \} ,$ ),

$$
\begin{array} { c } { { ( x _ { 1 } + ( N - 1 ) x _ { 2 } ) ^ { \prime } = \displaystyle \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } C _ { i } + ( 1 - \theta ) \sum _ { i = 1 } ^ { K } A _ { i } - \theta K Q + \theta \sum _ { i = 1 } ^ { K } A _ { i } + \theta K Q \right] } } \\ { { = \displaystyle \frac { \alpha } { N } \left[ G + \sum _ { i = 1 } ^ { K } ( C _ { i } + A _ { i } ) \right] , } } \end{array}\tag{232}
$$

independently of $\theta ;$ the same identity therefore holds for the Filippov boundary field at $u = 0$ . Hence

$$
X ^ { \prime } = \frac { \alpha } { N ^ { 2 } } \left[ G + \sum _ { i = 1 } ^ { K } ( C _ { i } + A _ { i } ) \right] .\tag{233}
$$

Because $E \le C _ { E }$ and $D _ { 2 , i } \geq e ^ { x _ { 1 } }$ , the $A _ { i }$ terms are $O ( e ^ { - x _ { 1 } } ) = O ( e ^ { - X } )$ . Moreover

$$
G + \sum _ { i = 1 } ^ { K } C _ { i } = e ^ { y _ { 1 } } \Big ( \frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } \Big ) + O ( e ^ { - x _ { 1 } } ) = a _ { 0 } ( t ) e ^ { Y - X } + O ( e ^ { - X } ) ,\tag{234}
$$

where, by Equation (231), $\begin{array} { r } { a _ { 0 } ( t ) = e ^ { X - x _ { 1 } } e ^ { x _ { 1 } } \big ( 1 / D _ { 1 } + \sum _ { i } 1 / D _ { 2 , i } \big ) } \end{array}$ is bounded above and below by positive constants once Y is large: $e ^ { X - x _ { 1 } } = e ^ { - \frac { N - 1 } { N } u } \in [ e ^ { - \frac { N - 1 } { N } C _ { u } } , 1 ]$ and $\begin{array} { r } { e ^ { x _ { 1 } } ( 1 / D _ { 1 } + \sum _ { i } 1 / D _ { 2 , i } )  } \end{array}$ $\begin{array} { r } { 1 + \sum _ { i = 1 } ^ { K } \frac { 1 } { K - i + 1 } } \end{array}$ . The $O ( e ^ { - X } )$ term is absorbed into the coeficient because $e ^ { - X } \leq e ^ { Y - X }$ for $Y \geq 0$ Thus, for all large t,

$$
X ^ { \prime } = a ( t ) e ^ { Y - X } ,\tag{235}
$$

with

$$
0 < a _ { - } \le a ( t ) \le a _ { + } < \infty .\tag{236}
$$

Next, writing $\eta \in [ 0 , 1 ]$ for the Filippov weight of $\mathbb { 1 } _ { v \geq 0 }$ (so $\eta = 1$ on $\{ v > 0 \}$ and $\eta = 0$ on $\{ v < 0 \} )$ , the $Y = y _ { 1 }$ equation is

$$
Y ^ { \prime } = \frac { \alpha } { N - 1 } \left[ K M + R - H - \sum _ { i = 1 } ^ { K } B _ { i } - \eta \sum _ { i = 1 } ^ { K } A _ { i } + ( 1 - \eta ) K Q \right] .\tag{237}
$$

Here

$$
K M = K \frac { N - 1 } { e ^ { Y } + N - 1 } = : b _ { 0 } ( t ) e ^ { - Y } ,\tag{238}
$$

with $b _ { 0 } ( t ) \in [ K ( N - 1 ) / 2 , K ( N - 1 ) ]$ once $e ^ { Y } \geq N - 1$ . Also

$$
H + \sum _ { i = 1 } ^ { K } B _ { i } = e ^ { Y } \Big ( \frac { 1 } { D _ { 1 } } + \sum _ { i = 1 } ^ { K } \frac { 1 } { D _ { 2 , i } } \Big ) = \widetilde { a } _ { 0 } ( t ) e ^ { Y - X } ,\tag{239}
$$

where $\begin{array} { r } { \widetilde { a } _ { 0 } ( t ) = e ^ { X } \big ( 1 / D _ { 1 } + \sum _ { i } 1 / D _ { 2 , i } \big ) } \end{array}$ is bounded above and below by positive constants for large $t ,$ by the same computation as for $a _ { 0 }$ . The remaining terms satisfy

$$
0 \le R \le ( N - K - 1 ) e ^ { - Y } , \qquad 0 \le ( 1 - \eta ) K Q \le K C _ { E } e ^ { - Y } ,\tag{240}
$$

$$
0 \leq \eta \sum _ { i } A _ { i } \leq C e ^ { - x _ { 1 } } \leq C e ^ { - 2 Y } ,\tag{241}
$$

using ${ \cal D } _ { 4 } \geq e ^ { Y } , { \cal E } \leq C _ { \cal E } , { \cal D } _ { 2 , i } \geq e ^ { x _ { 1 } }$ and Equation (231). Therefore

$$
Y ^ { \prime } = b ( t ) e ^ { - Y } - \widetilde a ( t ) e ^ { Y - X } + r ( t ) , \qquad r ( t ) : = \frac { \alpha } { N - 1 } \Big [ R + ( 1 - \eta ) K Q - \eta \sum _ { i } { \cal A } _ { i } \Big ] ,\tag{242}
$$

where $\begin{array} { r } { b ( t ) = \frac { \alpha } { N - 1 } b _ { 0 } ( t ) , \widetilde { a } ( t ) = \frac { \alpha } { N - 1 } \widetilde { a } _ { 0 } ( t ) } \end{array}$ satisfy

$$
0 < b _ { - } \leq b ( t ) \leq b _ { + } < \infty , \qquad 0 < \widetilde { a } _ { - } \leq \widetilde { a } ( t ) \leq \widetilde { a } _ { + } < \infty ,\tag{243}
$$

and

$$
| r ( t ) | \leq C e ^ { - Y } .\tag{244}
$$

We now check that the efective input $\beta ( t ) = b ( t ) + e ^ { Y } r ( t )$ is eventually bounded above and below by positive constants. We have

$$
e ^ { Y } r = { \frac { \alpha } { N - 1 } } \Bigl [ { \frac { ( N - K - 1 ) e ^ { Y } } { D _ { 4 } } } + ( 1 - \eta ) { \frac { K E e ^ { Y } } { D _ { 4 } } } - \eta e ^ { Y } \sum _ { i } { \cal A } _ { i } \Bigr ] .
$$

The first two terms lie in [0, $N - K - 1 + K C _ { E } ]$ , since $D _ { 4 } \geq e ^ { Y }$ and $1 \leq E \leq C _ { E } .$ ; they are of the same order as $b ,$ but they are nonnegative. For the last term, $\begin{array} { r } { e ^ { Y } \sum _ { i } A _ { i } \leq C e ^ { Y - x _ { 1 } } = C Z e ^ { - Y } \leq C e ^ { - Y }  0 } \end{array}$ by the bound $Z \leq C$ . Since $\begin{array} { r } { b ( t ) \geq \frac { \alpha } { N - 1 } \frac { K \left( N - 1 \right) } { 2 } = \frac { \alpha K } { 2 } } \end{array}$ once $e ^ { Y } \geq N - 1$ , we obtain

$$
\frac { \alpha K } { 4 } \leq \beta ( t ) \leq \alpha K + \frac { \alpha } { N - 1 } \big ( N - K - 1 + K C _ { E } \big )\tag{245}
$$

for all $t \geq t _ { 0 }$ , where $t _ { 0 }$ is chosen, using $Y  \infty$ , so that all the above estimates hold for $t \geq t _ { 0 }$

All hypotheses of Theorem C.4 are therefore satisfied on $[ t _ { 0 } , \infty )$ (the equations hold a.e. and for every Filippov selection, since all bounds are uniform in $\theta , \eta \in [ 0 , 1 ] )$ . Applying the lemma to $\widetilde X ( t ) = X ( t + t _ { 0 } ) , \widetilde Y ( t ) = Y ( t + t _ { 0 } )$ and using log $( t + t _ { 0 } ) = \log t + O ( 1 )$ for $t \geq 1$ gives

$$
X ( t ) = 2 \log t + O ( 1 ) , \quad \quad Y ( t ) = \log t + O ( 1 ) .\tag{246}
$$

Since $X = x _ { 1 } + O ( 1 ) = x _ { 2 } + O ( 1 )$ and $Y = y _ { 1 }$ , we obtain

$$
x _ { 1 } ( t ) , x _ { 2 } ( t ) = 2 \log t + O ( 1 ) , \qquad y _ { 1 } ( t ) = \log t + O ( 1 ) .\tag{247}
$$

Concerning the dependence on α: if $( x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } )$ solves the system with learning rate $\alpha ,$ then $t \ \mapsto \ ( x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } ) ( t / \alpha )$ solves the system with $\alpha \ = \ 1$ from the same initial state. Hence $( x _ { 1 } , x _ { 2 } , y _ { 1 } , y _ { 2 } ) ( t ) \ = \ ( { \hat { x } } _ { 1 } , { \hat { x } } _ { 2 } , { \hat { y } } _ { 1 } , { \hat { y } } _ { 2 } ) ( \alpha t )$ , where the hats denote the solution for $\alpha = 1$ , and all the $O ( 1 )$ terms above can be taken independent of α when the growth is expressed in terms of αt:

$$
x _ { 1 } ( t ) , x _ { 2 } ( t ) = 2 \log ( \alpha t ) + O ( 1 ) , \qquad y _ { 1 } ( t ) = \log ( \alpha t ) + O ( 1 ) .\tag{248}
$$

Since $y _ { 1 } - y _ { 2 } \leq C ,$ , hence

$$
y _ { 2 } ( t ) \geq y _ { 1 } ( t ) - C = \log ( \alpha t ) + O ( 1 ) .\tag{249}
$$

For the matching upper bound on $y _ { 2 } .$ , we use an elementary barrier. Let

$$
m ( t ) = \operatorname* { m a x } _ { 0 \leq r \leq t } y _ { 1 } ( r ) , \qquad h ( t ) = ( y _ { 2 } ( t ) - m ( t ) ) _ { + } .\tag{250}
$$

Both $y _ { 2 }$ and m are Lipschitz (the vector field is bounded), hence so is h. At almost every time at which $h ( t ) > 0$ , we have $y _ { 2 } ( t ) > m ( t ) \geq y _ { 1 } ( t )$ , hence $v < 0$ strictly, so the state is in the open region $\{ v < 0 \}$ where $y _ { 2 } ^ { \prime } = 0$ for every Filippov selection, while $m ^ { \prime } ( t ) \geq 0$ . Therefore $h ^ { \prime } ( t ) \leq 0$ for almost every such time, and consequently h is non-increasing (if $h ( t _ { 2 } ) > h ( t _ { 1 } )$ for some $t _ { 1 } < t _ { 2 }$ , let $t _ { * } = \operatorname* { s u p } \{ t \in [ t _ { 1 } , t _ { 2 } ] : h ( t ) \leq h ( t _ { 1 } ) \}$ ; then $h > h ( t _ { 1 } ) \geq 0$ on $( t _ { * } , t _ { 2 } ]$ and $\begin{array} { r } { h ( t _ { 2 } ) = h ( t _ { * } ) + \int _ { t _ { * } } ^ { t _ { 2 } } h ^ { \prime } \leq h ( t _ { 1 } ) } \end{array}$ a contradiction). Hence $y _ { 2 } ( t ) \leq m ( t ) + h ( 0 )$ , where $h ( 0 ) = ( y _ { 2 } ( 0 ) - y _ { 1 } ( 0 ) ) _ { + }$ is a constant (it vanishes for the trajectory of Theorem C.2, along which $v > 0$ at the initial time). Thus

$$
y _ { 2 } ( t ) \leq \operatorname* { m a x } _ { 0 \leq r \leq t } y _ { 1 } ( r ) + h ( 0 ) \leq \log ( \alpha t ) + O ( 1 ) \qquad { \mathrm { f o r ~ } } t \geq t _ { 0 } ,\tag{251}
$$

where we used that $y _ { 1 } ( r ) \leq \log ( \alpha r ) + C \leq \log ( \alpha t ) + C$ for $t _ { 0 } \leq r \leq t$ and that $y _ { 1 }$ is bounded on $[ 0 , t _ { 0 } ]$ . Therefore

$$
y _ { 2 } ( t ) = \log ( \alpha t ) + O ( 1 ) .\tag{252}
$$

This completes the proof.

By Theorem C.5, we can prove Theorem 4.2.

Proof of Theorem 4.2. By Theorems C.2, C.3 and C.5, for $t \geq T _ { 0 }$ we have

$$
a _ { 1 } \equiv c _ { 0 } , \qquad ( b _ { 2 } ) _ { + } = ( b _ { 2 } - b _ { 1 } ) _ { + } = 0 , \qquad d _ { 2 } \leq 0 ,\tag{253}
$$

$$
a _ { 2 } , b _ { 1 } = 2 \log ( \alpha t ) + O ( 1 ) , \qquad g _ { 1 } - d _ { 1 } = \log ( \alpha t ) + O ( 1 ) , \qquad g _ { 2 } - d _ { 2 } = \log ( \alpha t ) + O ( 1 ) ,\tag{254}
$$

and $0 \leq u = a _ { 2 } - b _ { 1 } \leq C _ { u } , v _ { + } = ( g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } ) _ { + } \leq C _ { v }$ for constants $C _ { u } , C _ { v }$ given by items 1 and 3 of Theorem C.5. The two-sided bound $| g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 } | \leq C$ follows from the last two growth rates for $t \geq t _ { 0 }$ and from continuity on $[ 0 , t _ { 0 } ]$ ; this proves the first part of the theorem. By Equation (129) the individual parameters satisfy

$$
g _ { 1 } , g _ { 2 } = \frac { N - 1 } { N } \log ( \alpha t ) + O ( 1 ) , \qquad d _ { 1 } , d _ { 2 } = - \frac { 1 } { N } \log ( \alpha t ) + O ( 1 ) ;\tag{255}
$$

in particular $d _ { 2 }  - \infty . \mathrm { ~ A l l }$ constants below depend only on $N , K , c _ { 0 } , c _ { 0 } ^ { \prime } .$

Logits on a general graph. Consider any directed graph on N vertices, any valid nonterminated DFS trace, and the current node $n _ { t }$ with stack parent $p \ ( { \mathrm { i f } } \ n _ { t } \neq n _ { \mathrm { { R o o t } } } )$ . We read the post-ReLU logits of Tables 1 and 2, using $b _ { 2 } = 0 , ( d _ { 2 } ) _ { + } = 0 ( \mathrm { s o } b _ { 2 } d _ { v } ^ { \mathrm { i n , t o t a l } } = 0$ and $( b _ { 2 } - b _ { 1 } m _ { v } ) _ { + } = 0 )$ $( g _ { 1 } - d _ { 1 } ) _ { + } = g _ { 1 } - d _ { 1 }$ and the dynamic readout, which sets the logit of $n _ { t }$ to 0 while keeping $n _ { t }$ in the softmax with weight 1:

• an unvisited child v: logit $( a _ { 2 } + b _ { 2 } ( d _ { v } ^ { \mathrm { i n , t o t a l } } - 1 ) ) _ { + } = a _ { 2 }$ , since $b _ { 2 } = 0$ for $t \geq T _ { 0 } ;$

• a visited child of $n _ { t }$ that is not the parent: logit at most $( a _ { 2 } - b _ { 1 } ) _ { + } + v _ { + } \leq u _ { + } + v _ { + } \leq C _ { u } + C _ { v } ;$

• the parent $p \mathrm { : }$ logit $( g _ { 1 } - d _ { 1 } + d _ { 2 } \mathbb { 1 } [ p = n _ { \mathrm { R o o t } } ] ) _ { + } + \mathbb { 1 } [ ( n _ { t }  p ) \in \mathcal { E } ] ( a _ { 2 } - b _ { 1 } m _ { p } ) _ { + }$ , which lies between $( g _ { 1 } - d _ { 1 } + d _ { 2 } \Im [ p = n _ { \mathrm { R o o t } } ] ) _ { + }$ and $g _ { 1 } - d _ { 1 } + u _ { + }$

• every other vertex, including the root when it is not the parent and $n _ { t }$ itself: logit 0.

Let m be the number of unvisited children, $m _ { 1 }$ the number of visited children other than the parent, and $m _ { 2 } \leq N$ the number of remaining vertices. The softmax normalizer Z therefore satisfies

$$
{ \mathcal { Z } } \ \leq \ m e ^ { a _ { 2 } } + m _ { 1 } e ^ { C _ { u } + C _ { v } } + e ^ { g _ { 1 } - d _ { 1 } + C _ { u } } + m _ { 2 } \ \leq \ m e ^ { a _ { 2 } } + C e ^ { g _ { 1 } - d _ { 1 } } + C .\tag{256}
$$

Case 1: $n _ { t }$ has unvisited children $( m \ge 1 )$ . The admissible set $C _ { t }$ consists of the unvisited children, each with logit $L _ { v } = a _ { 2 }$ , and the normalizer is $\begin{array} { r } { \mathcal { Z } = \sum _ { v \in C _ { t } } e ^ { L _ { v } } + \mathcal { Z } ^ { \prime } } \end{array}$ with $\mathcal { Z } ^ { \prime } \leq C e ^ { g _ { 1 } - d _ { 1 } } + C$ by Equation (256). Hence

$$
- \log \biggl ( \sum _ { v \in C _ { t } } p _ { \theta } ( v ) \biggr ) = \log \biggl ( 1 + \frac { \mathcal { Z ^ { \prime } } } { \sum _ { v \in C _ { t } } e ^ { L _ { v } } } \biggr ) \leq \log \biggl ( 1 + \frac { C e ^ { g _ { 1 } - d _ { 1 } } + C } { m e ^ { a _ { 2 } } } \biggr )\tag{257}
$$

$$
\leq 2 C e ^ { g _ { 1 } - d _ { 1 } - a _ { 2 } } = O \big ( ( \alpha t ) ^ { - 1 } \big ) ,\tag{258}
$$

where we used $g _ { 1 } - d _ { 1 } > 0$ (so $C \leq C e ^ { g _ { 1 } - d _ { 1 } } )$ and $a _ { 2 } - ( g _ { 1 } - d _ { 1 } ) = \log ( \alpha t ) + O ( 1 )$

Case 2: no unvisited child and $p \neq n _ { R o o t }$ . The admissible set is {p}. Writing $L _ { p }$ for the parent’s logit, $L _ { p } \geq g _ { 1 } - d _ { 1 }$ , and the normalizer is $e ^ { L _ { p } } + \mathcal { Z } ^ { \prime \prime }$ with ${ \mathcal { Z } } ^ { \prime \prime } \leq m _ { 1 } e ^ { C _ { u } + C _ { v } } + m _ { 2 } \leq C$ (the parent’s own term, including its forward contribution when $( n _ { t } \to p ) \in \mathcal { E }$ , is in the numerator as well). Hence

$$
- \log p _ { \theta } ( p ) = \log \Bigl ( 1 + \frac { \mathcal { Z } ^ { \prime \prime } } { e ^ { L _ { p } } } \Bigr ) \leq \log \bigl ( 1 + C e ^ { - ( g _ { 1 } - d _ { 1 } ) } \bigr ) = O \bigl ( ( \alpha t ) ^ { - 1 } \bigr ) .\tag{259}
$$

Case 3: no unvisited child and $p = n _ { R o o t }$ . Now the parent’s logit is $L _ { p } \geq g _ { 1 } - d _ { 1 } + d _ { 2 }$ , and the same computation gives

$$
- \log p _ { \theta } ( p ) \leq \log \bigl ( 1 + C e ^ { - ( g _ { 1 } - d _ { 1 } + d _ { 2 } ) } \bigr ) = O \Bigl ( e ^ { - \frac { N - 1 } { N } \log ( \alpha t ) } \Bigr ) = O \bigl ( ( \alpha t ) ^ { - \frac { N - 1 } { N } } \bigr ) ,\tag{260}
$$

since $\begin{array} { r } { g _ { 1 } - d _ { 1 } + d _ { 2 } = \frac { N - 1 } { N } \log ( \alpha t ) + O ( 1 ) } \end{array}$ . This rate cannot be improved: the masked current node contributes 1 to the normalizer, so − log $p _ { \theta } ( p ) \geq \log ( 1 + e ^ { - L _ { p } } )$ with $L _ { p } = g _ { 1 } - d _ { 1 } + d _ { 2 }$ exactly when $( n _ { t } \to n _ { \mathrm { R o o t } } ) \notin \mathcal { E } ,$ , and the graph with the single edge $( n _ { \mathrm { R o o t } } \to a )$ and the trace $n _ { \mathrm { R o o t } } , a , n _ { \mathrm { R o o t } }$ realizes this case.

A valid DFS trace on N vertices has at most $2 ( N - 1 )$ steps, and in every step one of the three cases applies (the terminated state, in which the root has no unvisited children, carries no loss term). Summing over the steps and taking the supremum over graphs and traces,

$$
\mathcal { L } _ { \mathrm { g e n } } \leq 2 ( N - 1 ) C ( \alpha t ) ^ { - \frac { N - 1 } { N } }\tag{261}
$$

for all $t \ge t _ { * } / \alpha$ , where $t _ { * }$ is such that all the $O ( 1 )$ estimates above are in force. This proves Equation (18). □

![](images/b36b6d0e0f8319da5146b7c8ec01184575e416ddc18f81cbe02be9fe6d7756eb.jpg)

![](images/c9427f6d4edc9d8dc856dfc8f2039270ec7e2cc7b237d3f75264e3f358b55152.jpg)  
Figure 2: Left: the individual parameters g , g (fitted slopes 0.935; predicted $( N - 1 ) / N = 0 . 9 3 7 5 )$ and $d _ { 1 } , d _ { 2 }$ (fitted slopes −0.062; predicted $- 1 / N )$ , and the conserved combinations $g _ { i } + ( N - 1 ) d _ { i }$ Right: the gap variables $u = a _ { 2 } - b _ { 1 } , v = g _ { 1 } - d _ { 1 } - g _ { 2 } + d _ { 2 }$ and $2 ( g _ { 1 } - d _ { 1 } ) - a _ { 2 }$ , which stay bounded as in Theorem C.5, together with $b _ { 2 }$ , which is frozen at 0 after the initial phase.

## C.2 Additional experimental results

Protocol. All runs use float64. The fixed first layer is implemented by its exact output on each token type (Equation (7)); the second layer, its value and output matrices, and the dynamic readout are those of Section 4, and the softmax is taken over the N vertex tokens. The reduced flow integrates the eight-scalar system of Theorem C.1 with an implicit solver in logarithmic time (relative tolerance 10<sup>−9</sup>), replacing each indicator $\mathbb { 1 } _ { x \geq 0 }$ by the steep ramp min{max $\{ x / \epsilon + 1 , 0 \} , 1 \}$ with $\epsilon = 1 0 ^ { - 6 } ;$ changing ϵ by two orders of magnitude moves the trajectory by $O ( \epsilon )$ , and plain gradient descent with $\eta = 0 . 0 5$ agrees with it to within 0.012 on $[ 1 , 5 \times 1 0 ^ { 5 } ]$ . Full-matrix training uses $B = 1 2 8$ random vertex permutations per gradient step, i.e., a stochastic version of the loss Equation (13); the efective scalars of a matrix are the mean of its diagonal and the mean of its of-diagonal entries. The implementation was checked against a direct autograd implementation, against finite diferences, against the closed-form node logits of Tables 1 and 2 on random graphs, and, for $N = 7 ,$ against the exact average over all 5040 permutations.

The test set contains, for N = 16, 300 random graphs (Erdős–Rényi graphs with edge probability in {0.1, 0.2, 0.4}, random out-trees with up to three extra edges, and “root-fan” graphs in which the root starts several chains and the target is unreachable or on a late chain), together with the full canonical trace and the single-edge graph $\{ ( n _ { \mathrm { R o o t } }  a ) \}$ . For each graph we enumerate all valid DFS prefixes (or sample 50 traces when there are more), giving 64,325 distinct states, of which 5,625 are forward steps at a node that has an edge back to its stack parent.

Individual parameters and gap variables. Figure 2 shows the quantities behind Equation (129) and Theorem C.5. The combination $g _ { 1 } + ( N - 1 ) d _ { 1 }$ is conserved to $5 \times 1 0 ^ { - 1 2 }$ along the reduced flow; $g _ { 2 } + ( N - 1 ) d _ { 2 }$ moves by 0.019 during the initial phase, while $d _ { 2 } > 0$ , and is constant afterwards, as predicted. The gap $u = a _ { 2 } - b _ { 1 }$ is nonnegative throughout and converges to about 4.8 after a transient maximum of 5.1 at $t \approx 4 0 0 ;$ v rises to 0.34 during the initial phase and then converges to 0, i.e., the trajectory slides on the switching surface $v = 0$ for all $t \gtrsim 1 0 ^ { 2 } \mathrm { { : } }$ and $2 ( g _ { 1 } - d _ { 1 } ) - a _ { 2 }$ converges to about 5.6.

![](images/e77bfd0dcd5b0f52fbc6c77cc1eb77ab6839ca5715d5e71f7eff2629d088c384.jpg)

![](images/dedcc3fdd618d7c26f8bc194421ae2f2dd30cef19e1fe8bdeb115f3c9a811783.jpg)  
Figure 3: Left: relative distance $\| X - ( \bar { x } _ { \mathrm { d i a g } } I + \bar { x } _ { \mathrm { o f f } } E _ { \mathrm { o f f } } ) \| _ { F } / \| X \| _ { F }$ of each trained matrix from the invariant subspace during full-matrix training with $B = 1 2 8$ sampled permutations per step. Right: fraction of test states at which the argmax is a valid DFS move, by step type.

Invariant subspace under sampled permutations. Theorem 4.1 concerns the exact permutation averaged loss. With B = 128 sampled permutations per step (Figure 3, left), the iterate leaves the subspace during the initial phase, when the dynamics slide along the switching surfaces $b _ { 2 } = 0$ and $v = 0 \colon$ the relative distance of $\scriptstyle { B _ { 1 } }$ and $B _ { 2 }$ reaches 0.7–0.9 and the efective scalars difer from the reduced flow by up to 0.9 at $t \approx 1 5$ . The iterate then returns: the distance is below $1 0 ^ { - 2 }$ from $t \approx 7 0$ on and at most $6 \times 1 0 ^ { - 4 } ~ \mathrm { a t } ~ t = 5 \times 1 0 ^ { 5 }$ , where the efective scalars agree with the reduced flow to within 0.03. The excursion is a sampling efect: rerunning the early phase with $B \in \{ 5 1 2 , 2 0 4 8 \}$ reduces it only partially, whereas $B \geq 8 1 9 2$ keeps the distance below 0.03 and the scalars within 0.007 of the reduced flow at all times, and reducing η tenfold at $B = 1 2 8$ has no efect.

Exponent versus N. The table in Section 5 reports the exponent of the root-backtracking loss fitted on $t \in [ 1 0 ^ { 6 } , 1 0 ^ { 9 } ]$ for $N \in \{ 8 , 1 0 , 1 2 , 1 6 , 2 4 , 3 2 \}$ ; every fit lies within $3 \times 1 0 ^ { - 4 } ~ \mathrm { o f } ~ - ( N - 1 ) / N$ Figure 4 shows the local log–log slope of the same curves: it approaches − $( N - 1 ) / N$ from below, and the transient lasts longer for larger N, because the $O ( 1 )$ terms in $\begin{array} { r } { d _ { 2 } = - \frac { 1 } { N } \log ( \alpha t ) + O ( 1 ) } \end{array}$ then take longer to become negligible.

Accuracy and back-edge states. Figure 3 (right) shows when the argmax becomes a valid DFS move. Backtracking steps are correct from $t \approx 5$ , forward steps without an edge back to the parent from $t \approx 4 0$ , and forward steps with such an edge only from $t \approx 1 . 8 \times 1 0 ^ { 4 }$ , when $b _ { 1 }$ exceeds $g _ { 1 } - d _ { 1 }$ ; the overall accuracy is 0.91 at $t = 1 0 0$ and equals 1 from $t \approx 2 \times 1 0 ^ { 4 } ~ \mathrm { o n }$ , and both training procedures give the same curves. The back-edge states are also the ones that attain the supremum $\mathcal { L } _ { \mathrm { g e n } }$ over the whole observed range: a visited child of the current node that is also its stack parent receives votes from both heads, $g _ { 1 } - d _ { 1 }$ from the backtracking head and $a _ { 2 } - b _ { 1 }$ from the forward head, so its logit gap to the correct unexplored child is $b _ { 1 } - ( g _ { 1 } - d _ { 1 } ) = \log ( \alpha t ) - u + O ( 1 )$ and the loss is $\Theta ( e ^ { u } / t )$ with $e ^ { u } \approx e ^ { 5 }$ . Both efects are consequences of the two-head mechanism itself rather than of the training horizon.

![](images/8e4b869f4f9b4f93cd7744137766c46e12a848c4096ab0ea2595ad01e0354d27.jpg)  
Figure 4: Local log–log slope d log $\mathcal { L } / \mathrm { d }$ log t of the root-backtracking loss for several N (reduced flow). Dashed lines mark the predicted exponents $- ( N - 1 ) / N$

Initialization. Figures 5 and 6 examine the role of the initialization Equation (16). The allconstant initialization $c _ { 0 } \pmb { { \cal E } }$ lies on the two switching surfaces $b _ { 2 } = b _ { 1 }$ and $g _ { 1 } = d _ { 1 }$ , where the vector field is discontinuous, and its behaviour depends on the convention for $\mathrm { R e L U ^ { \prime } ( 0 ) }$ and on $( N , K )$ With ReL $. \mathrm { U } ^ { \prime } ( 0 ) = 0$ (the default of common autograd libraries) and exactly cancelling head sums, $g _ { 1 } - d _ { 1 }$ remains exactly 0 and the backtracking head never activates; the forward parameters then grow only half as fast $( a _ { 2 } , b _ { 1 } \approx 1 1 . 5$ at $t = 5 \times 1 0 ^ { 5 }$ against 22.0 and 17.2). With $\mathrm { R e L U ^ { \prime } } ( 0 ) = 1$ the outcome depends on $( N , K ) { \mathrm { : } }$ at (16, 8) the trajectory enters the region $\left\{ b _ { 2 } < b _ { 1 } , \ g _ { 1 } > d _ { 1 } \right\}$ at once, but at (16, 2) and (40, 3) it first exits towards $b _ { 2 } > b _ { 1 }$ with $b _ { 1 } < 0$ for a transient of several hundred time units, after which it approaches the trajectory from Equation (16) in $a _ { 2 } , b _ { 1 } , g _ { 1 } - d _ { 1 }$ but keeps a positive $b _ { 2 }$ that decays only logarithmically slowly (0.34 at $t = 5 \times 1 0 ^ { 5 }$ for (16, 2)), its gradient being $O ( 1 / t )$ . Across 60 pairs $( N , K )$ with $8 \leq N \leq 6 4$ and $2 \leq K \leq 1 0$ , the wrong-side exit occurs in all but one case in which $N > ( K + 2 ) ^ { 2 } / 2$ , and in none of the others. In practice, floating-point round-of or permutation sampling is enough to release the frozen case (Figure 5, D3 with round-of and D4), but the trajectory is then selected by numerical noise rather than by the gradient flow. The initialization Equation (16) avoids all of these efects: it starts inside the region that Theorem C.2 shows to be invariant and stays there from $t = 0$ in every setting we tried.

Summary. The experiments agree with the training-dynamics theory in every quantitative prediction: the forward-head parameters grow at 2 log t, the backtracking separation at log t, the individual parameters at $\textstyle { \frac { N - 1 } { N } }$ log t and $- { \frac { 1 } { N } }$ log t, the gap variables remain bounded, and backtracking to the root decays at exactly $t ^ { - ( N - 1 ) / N }$ for N between 8 and 32. They also show which parts of the statements are asymptotic: the supremum loss is dominated over all practical horizons by $t ^ { - 1 }$ terms with large constants, and sampled-permutation training approaches the invariant subspace only after the initial phase.

![](images/cb83f57012cfb00bbe411b87f42f90f1b01795ed9396da6e7e239dbe573d3fb4.jpg)

![](images/6770213c265dcfa8e34d7e47e9d5e861c01bfa2e56d9e076bac0c90e919ae563.jpg)

![](images/663353adcd1f11319b9370e7206c1366cd2272b14b94239f8917362c68436568.jpg)

![](images/44114a28e33919d07b7795081158acb78fff9b385961ef7bdd247f6998cd204c.jpg)  
D1: proposed initialization D3: constant, ReLU<sup>′</sup>(0) = 0 D4: full matrices (3 seeds) D2: constant, ReLU<sup>′</sup>(0) = 1 D3 with round-off

Figure 5: Initialization ablation at $N = 1 6 , K = 8$ (reduced flow by gradient descent with $\eta = 0 . 0 5 ;$ D4 is full-matrix training with three seeds). D1: the initialization Equation (16). D2, D3: the all-constant initialization $c _ { 0 } \pmb { { \cal E } }$ with the conventions $\mathrm { R e L U } ^ { \prime } ( 0 ) ~ = ~ 1$ and ReL $\mathrm { { { U } ^ { \prime } ( 0 ) = \ 0 } }$ . With ReL $. \mathrm { U } ^ { \prime } ( 0 ) = 0$ and exactly cancelling head sums, $g _ { 1 } - d _ { 1 }$ stays at 0 forever and the backtracking head never activates; floating-point round-of in the head sums, or sampling noise (D4), releases it.

![](images/f3a2f29489f95a4713a8b742da5e418068bd565fad05eb127a0911bb43880b9f.jpg)  
Figure 6: The all-constant initialization $c _ { 0 } \pmb { { \cal E } }$ with $\mathrm { R e L U ^ { \prime } ( 0 ) } = 1 \mathrm { a t } ( N , K ) = ( 1 6 , 2 ) , ( 4 0 , 3 )$ , and (16, 8). When N is large relative to $K .$ , the trajectory first leaves the switching surface $b _ { 2 } = b _ { 1 }$ towards $b _ { 2 } > b _ { 1 }$ and $b _ { 1 }$ becomes negative, before re-entering the region $\{ b _ { 2 } < b _ { 1 } , \ g _ { 1 } > d _ { 1 } \}$ at $t \approx 3 0 0$ and $t \approx 1 0 ^ { 3 }$ , respectively; at (16, 8) it enters the region immediately. The initialization Equation (16) stays in the region from $t = 0$ in all three cases.