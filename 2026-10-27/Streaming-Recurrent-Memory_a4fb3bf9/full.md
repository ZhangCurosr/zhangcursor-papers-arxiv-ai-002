## Streaming Recurrent Memory

Mohamed Elsayed Department of Computing Science University of Alberta

Minh Pham   
Department of Computing Science   
University of Alberta

A. Rupam Mahmood Department of Computing Science University of Alberta CIFAR Canada AI Chair, Amii

mohamedelsayed@ualberta.ca

minh.pham@ualberta.ca

armahmood@ualberta.ca

## Abstract

Natural intelligence learns from a continuous stream of sensorimotor experience: data can arrive at high bandwidth, and behavior adapts to feedback as soon as it arrives, even when it is delayed. In contrast, most modern deep learning systems rely on stored datasets, replay bufers, and retained contexts through which credit is later assigned. At high data rates, storing and revisiting past experience can be impractical under tight memory and computational constraints. This paper studies recurrent memory under a streaming learning constraint: the learner learns from experience as soon as it arrives without storing raw experience. We ask whether a streaming learner can maintain a state that remains useful for future prediction and control from delayed feedback. We introduce Streaming Recurrent Memory (SRM), a recurrent memory designed for the streaming setting which preserves information that may matter later while remaining trainable with one-step backpropagation through time (BPTT). SRM provides a streaming-compatible recurrence that is designed to recall past information by content or time, track elapsed steps, and compose transforma tions whose outcome depends on the order in which inputs are observed. Across controlled memory tasks and streaming applications, SRM learns efectively with one-step BPTT, outperforming recurrent baselines in associative recall and online language modeling while achieving near-perfect success in associative T-Maze with delayed rewards. These results suggest that fixing how inputs are stored in the recurrent state, while learning how that state is read out, enables efective streaming learning without long-range backpropagation.

## 1 Introduction

Many natural data sources are streams: text, audio, video, and sensorimotor experience unfold continuously, with temporal structure spanning arbitrarily long intervals. Modern sequence learners, particularly Transformers (Vaswani et al. 2017), are instead typically trained on shufled collections of stored segments. This strategy exploits parallel hardware and underlies state-of-the-art results across language and vision (Brown et al. 2020, Hofmann et al. 2022, Touvron et al. 2023), but does not preserve the original data stream. A natural alternative is to learn directly from the stream itself: experience arrives once, in order, and the learner must learn without relying on stored history. This setting is not only close to how natural agents learn from continuous, non-repeatable experience, but also matches the constraints faced by many deployed systems, including real-time control and robotics (Vasan et al. 2024, Verma et al. 2023), never-ending video prediction (Carreira et al. 2024, Qian et al. 2024, Han et al. 2025), continual learning (Yoo et al. 2024),

![](images/c9b363de2c77618b3dce72372e3cdb962daf611d5e1b5cf8234309aec39bbb3e.jpg)

Figure 1: The streaming learner. The input and feedback arrive sequentially, and the learner updates based on each sample as it arrives without storing. Feedback ${ \mathbf { } } _ { \mathbf { } } \mathbf { \cdot } \mathbf { } \mathbf { \sigma } _ { \mathbf { } }$ may be instructive, such as labels or targets in supervised learning, or evaluative, such as rewards in reinforcement learning, and often arrives delayed. The output $\mathbf { } _ { o _ { t } }$ can be a prediction or an action. The learning algorithm updates the parameterized recurrent dynamics function $f _ { \theta }$ and the parameterized output function $g _ { \phi }$

and adaptive interaction (Dai et al. 2025). We study this setting using recurrent learners, which process one input at a time while maintaining a state that carries information from past inputs into future computations and is used to produce predictions or actions. Figure 1 illustrates the streaming setting studied in this paper: inputs and feedback arrive sequentially, where the learner updates its recurrent state, which carries information from the past, and learns using the provided feedback and its current input.

Learning directly from a stream requires more than compressing a long history into a non-growing state (Hochreiter & Schmidhuber 1997). At each moment, the learner does not yet know which parts of the past will matter: it may later need to recall an observation, retrieve a value associated with a key, estimate elapsed time, reset a learned context, track a state progression, or assign delayed feedback to the state that produced it (Sutton 1988, Schmidhuber 1992, Graves et al. 2014, Weston et al. 2015, Sukhbaatar et al. 2015, Schlag et al. 2021, Merrill et al. 2024). For example, a robot navigating a room may need to remember where it saw an object, bind that object to a location, track its pose, reset context when entering a new room, and credit earlier decisions for later success or failure (Parisotto & Salakhutdinov 2017). Thus, the recurrent state must both preserve useful structure from the past and make that structure usable by later learning signals. Both requirements must be satisfied for efective learning; a memory whose update is easy to learn may not preserve the information needed later, while a memory that preserves information may still be useless if delayed feedback cannot shape how it is used. The streaming setting, therefore, places a stronger constraint on recurrent architectures as their states must not only be expressive enough to store the right structure, but also be learnable from feedback observed in the stream, even when delayed.

Modern recurrent and state-space networks are a natural starting point because they maintain a non-growing state that can be updated step by step (e.g., Orvieto et al. 2023, Gu & Dao 2024, De et al. 2024). However, streaming learning restricts how a recurrent state update can be learned. In ofline sequence learning, the learner can learn how information is written into the recurrent state through gradients propagated across a retained window using backpropagation through time (BPTT). If information written at one time is useful later, the later feedback can shape the earlier writing mechanism to incorporate that information. Streaming learning rules out this retained window. Real-Time Recurrent Learning (RTRL; Williams & Zipser 1989) can recover these gradients online, but it must store and update the sensitivity of the state to every parameter, which is tractable only for restricted recurrences, and in practice, it is only approximate when recurrent layers are stacked (Zucchet et al. 2023), defeating its goal of computing exact gradients. In this work, we study streaming learning with BPTT truncated to one step instead. Under one-step BPTT, the current feedback can train how to use the current state, but it cannot directly correct the specific past update that produced that state. As a result, architectures whose useful memory depends on a learned writing mechanism face a credit-assignment bottleneck. This setting, therefore, calls for a diferent design principle: the state should include as much information as possible, while learning should act mainly on how that state is read.

We introduce Streaming Recurrent Memory (SRM), a recurrent unit built around this principle. SRM constructs memory with a fixed writing mechanism that stores each input in a structured content–time address space instead of learning what to write from later feedback, a gradient path absent in one-step BPTT. The learner is focused on how to read from the memory, where gradients are available immediately: the state preserves information as the stream arrives, while the learner learns how to address and use that state for prediction or control. We evaluate SRM through capability probes, credit-assignment diagnostics, controlled memory tasks, and streaming applications, including next-byte language modeling and streaming deep reinforcement learning. Overall, this paper makes three contributions: we identify why streaming learning with one-step BPTT favors a memory whose writing mechanism does not depend on delayed credit, we introduce a content–time associative memory with a separate tracking channel for order-sensitive computation, and we show that this design learns memory-dependent behavior under one-step BPTT across controlled tasks, reinforcement learning, and language modeling.

## 2 Background

Let us consider a recurrent system with dynamics that can be written as ${ \pmb h } _ { t } = f ( { \pmb h } _ { t - 1 } , { \pmb x } _ { t } , \pmb \theta )$ , where $\boldsymbol { h } _ { t } \in \mathbb { R } ^ { n }$ is the learner’s state vector, $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$ is the input vector, and $\pmb \theta = \{ \pmb { W } _ { 1 } , \dots , \pmb { W } _ { p } \}$ is the set of learnable parameters of the recurrence function, where each $W \in \theta$ is a weight matrix. The output is given by $\hat { \pmb { y } } _ { t } = g ( \pmb { h } _ { t } , \pmb { x } _ { t } , \phi )$ , where $\hat { \pmb y } _ { t } \in \mathbb { R } ^ { m }$ , and $\phi = \{ V _ { 1 } , \ldots , V _ { q } \}$ is the set of learnable parameters of the output function, where each $V \in \phi$ is a weight matrix. To learn the parameters, we need to compute $\nabla { \mathcal { L } } _ { W } , \ \forall W \in \theta$ and $\nabla { \mathcal { L } } _ { V } , \ \forall V \in \phi$ . Given a target $\pmb { y } _ { t } \in \mathbb { R } ^ { m }$ , the gradient of the loss with respect to $W .$ , treating it as a flattened vector, is given by $\begin{array} { r } { \frac { \partial \mathcal { L } ( y _ { t } , \hat { y } _ { t } ) } { \partial W } = \left( \frac { \mathrm { d } h _ { t } } { \mathrm { d } W } \right) ^ { \top } \frac { \partial \mathcal { L } } { \partial h _ { t } } , \quad \forall W \in \pmb { \theta } , } \end{array}$

## 2.1 Backpropagation Through Time (BPTT)

Note that to obtain $\frac { \mathrm { d } h _ { t } } { \mathrm { d } W }$ , we must account for both the direct and indirect efects of changing $W$ on $\boldsymbol { h } _ { t } .$ since W influences $\boldsymbol { h } _ { t }$ directly and also through every earlier state. Since we can unroll the function $\boldsymbol h _ { t } = f ( f ( f ( \dots f ( h _ { 0 } , x _ { 1 } , \theta ) , \dots x _ { t - 2 } , \theta ) , x _ { t - 1 } , \theta ) , x _ { t } , \theta )$ , we write the gradient $\frac { \mathrm { d } h _ { t } } { \mathrm { d } W }$ as

$$
\frac { \mathrm { d } \pmb { h } _ { t } } { \mathrm { d } \pmb { W } } = \frac { \partial \pmb { h } _ { t } } { \partial \pmb { W } } + \frac { \partial \pmb { h } _ { t } } { \partial \pmb { h } _ { t - 1 } } \frac { \mathrm { d } \pmb { h } _ { t - 1 } } { \mathrm { d } \pmb { W } } , \quad \forall \pmb { W } \in \pmb { \theta } .\tag{1}
$$

We need to keep unrolling further because W again afects $\boldsymbol { h } _ { t - 1 }$ through two pathways. Let us define $\begin{array} { r } { S _ { t } \doteq \frac { \mathrm { d } h _ { t } } { \mathrm { d } W } , J _ { t } \doteq \tilde { \frac { \partial h _ { t } } { \partial W } } } \end{array}$ , and $\begin{array} { r } { \bar { B } _ { t } \doteq \frac { \partial h _ { t } } { \partial h _ { t - 1 } } } \end{array}$ . We can write the recursive relationship as follows:

$$
{ \begin{array} { l } { S _ { t } = J _ { t } + B _ { t } S _ { t - 1 } } \\ { \displaystyle \quad = J _ { t } + B _ { t } J _ { t - 1 } + B _ { t } B _ { t - 1 } S _ { t - 2 } } \\ { \displaystyle \quad = \sum _ { j = 1 } ^ { t } \left( \prod _ { i = j + 1 } ^ { t } B _ { i } \right) J _ { j } . \qquad { \mathrm { ( u n d e r ~ t h e ~ a s s u m p t i o n ~ t h a t ~ } } h _ { 0 } { \mathrm { ~ i s ~ i n d e p e n d e n t ~ o f ~ } } W { \mathrm { ) } } } \end{array} }\tag{2}
$$

This relationship is utilized in BPTT by eficiently calculating the summation backward without forming or storing $S _ { t }$ . Calculating this requires storing all previous inputs and states $\pmb { x } _ { i } , \pmb { h } _ { i } , \forall \{ 1 , \dots , t \}$ . In other words, we need to backpropagate the gradient in time, starting from the current time step and going all the way back to the first time step. The computational and memory cost, therefore, grows linearly with the sequence length, since each update must traverse the entire history.

We can simplify the intensive computation needed by BPTT if we truncate the backpropagation process by going back in time up to time step $t - T$ , where $t > T$ . This creates the Truncated-Backpropagation Through Time (T-BPTT) gradient (Williams & Peng 1990):

$$
{ \pmb S } _ { t } \approx \sum _ { j = t - T } ^ { t } \left( \prod _ { i = j + 1 } ^ { t } B _ { i } \right) { \pmb J } _ { j } , \quad \forall t > T .\tag{3}
$$

T-BPTT reduces the cost of full BPTT by retaining only the most recent $T + 1$ inputs and states. Its approximation, however, removes all credit-assignment paths that extend more than $T$ steps into the past, which can make learning myopic. It also requires a finite bufer of past experience, and is therefore not streaming unless $T = 1$ . We refer to this limiting case as one-step $\mathrm { B P T T }$ . Although one-step BPTT propagates gradient credit only one step backward, it can still learn dependencies over longer horizons indirectly, because the recurrent state evolves online and may preserve information from earlier observations. Algorithm 1 in Appendix A gives the one-step BPTT pseudocode.

## 2.2 Real-time Recurrent Learning (RTRL)

Instead of relying on the rolled-out equation (Eq. 2 or $\operatorname { E q . 3 } )$ , we can instead compute $\begin{array} { r } { S _ { t } = \frac { \mathrm { d } h _ { t } } { \mathrm { d } W } } \end{array}$ , known also as the sensitivity matrix, incrementally using its recursive relationship. This process is known as Real-Time Recurrent Learning (RTRL) (Williams & Zipser 1989) given by

$$
\begin{array} { r } { S _ { t } = J _ { t } + B _ { t } S _ { t - 1 } . } \end{array}\tag{4}
$$

The quantities $J _ { t }$ and $B _ { t }$ depend only on the current input and state, so they can be computed without BPTT; only $S _ { t - 1 }$ must be stored to update $S _ { t }$ incrementally. Unlike T-BPTT, RTRL computes the exact gradient without approximation, provided the parameters are held fixed. In general, storing and updating $S _ { t }$ is computationally expensive and scales poorly with the number of units and parameters, but RTRL becomes practical mainly when the recurrence has an exploitable structure, such as diagonal or sparse dynamics. These savings, however, hold only for a single recurrent layer. When stacked, the input to an upper layer depends on the parameters and past states of the layers below it, so exact sensitivities must also track these cross-layer dependencies and lose their elementwise structure. BPTT does not sufer from this limitation since it never stores $S _ { t }$ . In practice, RTRL is applied within each layer, treating the output of the layer below as a fixed input, which makes the gradient approximate for the multi-layer networks case (Zucchet et al. 2023). Algorithm 2 in Appendix B gives the RTRL pseudocode.

## 2.3 State Construction

The recurrent update determines both what enters the state and how existing content is transformed. We focus on recurrences that are linear in the state, the class used by modern linear recurrent and state-space models (Gu et al. 2022, Orvieto et al. 2023, Gu & Dao 2024, De et al. 2024). Linearity in the state separates these two roles explicitly and makes stability a property of the transition alone. Any such recurrence can be written as

$$
\pmb { h } _ { t } = \pmb { A } ( \pmb { x } _ { t } ) \pmb { h } _ { t - 1 } + \pmb { b } ( \pmb { x } _ { t } ) .\tag{5}
$$

We call $b : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ the write: it maps the current input to the new content added to the state. We call $\pmb { A } : \mathbb { R } ^ { d }  \mathbb { R } ^ { n \times n }$ the transition: it maps the current input to the matrix that transforms the content already in the state. Both may carry learnable parameters, and we here make no further assumption on either.

Input-independent transition. When the transition is a fixed matrix, $\mathbf { \delta } _ { \mathbf { { A } } ( \mathbf { \delta } _ { \mathbf { { x } } } ) } = \mathbf { \delta } _ { A }$ for every input ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ unrolling Eq. 5 from $\mathbf { { \boldsymbol { h } } } _ { 0 } = \mathbf { { \boldsymbol { 0 } } }$ gives $\begin{array} { r } { \pmb { h } _ { t } = \sum _ { s = 1 } ^ { t } \pmb { A } ^ { t - s } \pmb { b } ( \pmb { x } _ { s } ) } \end{array}$ . Each write is therefore transformed according to its age. This supports retention, decay, oscillation, and fixed delays, but the current input cannot change how existing memory is transformed.

Modern linear recurrent and state-space layers exploit this structure directly. For example, LRU and S5 parameterize the fixed transition as a diagonal matrix with complex entries (Orvieto et al. 2023, Smith et al. 2023): the magnitude of each entry sets how quickly the corresponding state coordinate decays, and its phase sets how fast it oscillates. The resulting traces distinguish when an input occurred, but their subsequent evolution is independent of which inputs arrive later.

Input-dependent transition. When the transition depends on the current input, unrolling Eq. 5 gives $\begin{array} { r } { \pmb { h } _ { t } = \sum _ { s = 1 } ^ { t } \left( \prod _ { r = s + 1 } ^ { t } \pmb { A } ( \pmb { x } _ { r } ) \right) \pmb { b } ( \pmb { x } _ { s } ) } \end{array}$ . Now, each write $\pmb { b } ( \pmb { x } _ { s } )$ is transformed by the ordered product of all past transitions. A common choice restricts the transition to be diagonal for every input, $\pmb { A } ( \pmb { x } ) = \mathrm { D i a g } ( \pmb { a } ( \pmb { x } ) )$ with $\pmb { a } : \mathbb { R } ^ { d }  \mathbb { R } ^ { n } \mathrm { ( o r } \mathbb { C } ^ { n } \mathrm { ) }$ . This choice provides useful input-dependent retention and forgetting, but any two such transitions commute: $A ( { \bf { x } } _ { i } ) A ( { \bf { x } } _ { j } ) = A ( { \bf { x } } _ { j } ) A ( { \bf { x } } _ { i } )$ . Consequently, exchanging two diagonal transitions does not change how they transform content already in the state. To make that transformation depend on order, the transition family must contain a non-commuting pair, $A ( x _ { i } ) A ( x _ { j } ) \neq A ( x _ { j } ) A ( x _ { i } )$ Such transitions support computations in which inputs act as ordered operations on memory, including finite-state, permutation, and group tracking (Merrill et al. 2024).

Input-dependent diagonal transitions are used by selective state-space and gated linear-recurrent models to control retention from the current input (Gu & Dao 2024, De et al. 2024, Katsch 2023, Qin et al.

2024). Replacing them with unrestricted dense matrices would permit non-commutativity but would give up the elementwise update and the simple stability constraint that motivates diagonal recurrences: stability would instead depend on controlling the global spectral properties of the transition matrix, while repeated unconstrained matrix products can produce vanishing or exploding dynamics (Pascanu et al. 2013, Miller & Hardt 2019, Zucchet & Orvieto 2024).

Householder transitions. A structured transition that does not commute, yet stays cheap and stable, is the generalized Householder transformation (Householder 1958, Grazzi et al. 2025), $\pmb { A } ( \pmb { x } _ { t } ) = \pmb { I } - \beta _ { t } \pmb { u } _ { t } \pmb { u } _ { t } ^ { \top }$ with $\| \mathbf { \boldsymbol { u } } _ { t } \| = 1$ . It leaves the part of the state orthogonal to the axis $\mathbf { \Delta } \mathbf { u } _ { t }$ unchanged and scales the part along it by $1 - \beta _ { t }$ . For $\beta _ { t } \in ( 0 , 2 ]$ it never expands the state, and at $\beta _ { t } = 2$ it is a norm-preserving reflection. Two transitions whose axes are neither parallel nor orthogonal do not commute, so the state depends on the order of the inputs. For example, with $\beta _ { t } = 2$ , choosing ${ \pmb u } _ { t } = ( { \pmb e } _ { i } - { \pmb e } _ { j } ) / \sqrt { 2 }$ swaps coordinates i and j. By choosing $\mathbf { \pmb { u } } _ { t }$ according to each successive swap, a one-hot state can track an object through an arbitrary sequence of swaps. With $\beta _ { t } \in ( 0 , 1 )$ , this is the transition of DeltaNet (Schlag et al. 2021, Yang et al. 2024b); extending $\beta _ { t }$ to (0, 2) lets linear RNNs track parity and permutations (Grazzi et al. 2025, Siems et al. 2025).

## 3 Streaming Recurrent Memory

We want a recurrent memory that can be trained with one-step BPTT and still compose inputs in an order-sensitive way. We design it around four desiderata. (D1) Reliable credit-free write: as one-step BPTT feedback cannot reach how earlier inputs were written, each input should enter the state in a form that can be retrieved later, without requiring the writing mechanism to be learned. (D2) Cheap computation: the recurrence should be structured such that its parameters can be learned cheaply. (D3) Order sensitivity: part of the state should evolve under input-dependent transitions that do not commute. (D4) Stability: repeated transitions should stay bounded by construction.

SRM specializes Eq. 5 into two states, a matrix memory $\pmb { H } _ { t } \in \mathbb { R } ^ { d _ { k } \times d _ { v } }$ with key width $d _ { k }$ and value width $d _ { v } .$ , and a tracking vector $z _ { t } \in \mathbb { R } ^ { G }$

$$
\begin{array} { r } { \mathbf { H } _ { t } = \mathbf { \boldsymbol { \Lambda } } ( \mathbf { \boldsymbol { x } } _ { t } ) \mathbf { \boldsymbol { H } } _ { t - 1 } + \mathbf { \boldsymbol { k } } _ { t } \mathbf { \boldsymbol { v } } _ { t } ^ { \top } , \qquad \mathbf { \boldsymbol { z } } _ { t } = \left( \mathbf { \boldsymbol { I } } - \beta _ { t } \mathbf { \boldsymbol { u } } _ { t } \mathbf { \boldsymbol { u } } _ { t } ^ { \top } \right) \mathbf { \boldsymbol { z } } _ { t - 1 } , \qquad \mathbf { \boldsymbol { r } } _ { t } = \left[ \mathbf { \boldsymbol { q } } _ { t } ^ { \top } \mathbf { \boldsymbol { H } } _ { t } \right] . } \end{array}\tag{6}
$$

The memory decays each row by the retention $\pmb { \Lambda } ( \pmb { x } _ { t } ) = \mathrm { d i a g } \big ( \exp ( - \exp ( \pmb { W } _ { \lambda } \pmb { x } _ { t } + \pmb { b } _ { \lambda } ) ) \big )$ and stores the value ${ \pmb v } _ { t } = { \pmb V } _ { x } { \pmb x } _ { t } + { \pmb V } _ { p } { \pmb x } _ { t - 1 }$ under the unit-norm key $k _ { t } = \bar { k } _ { t } / \| \bar { k } _ { t } \|$ , with $\bar { \pmb { k } } _ { t } = \pmb { K } _ { x } \pmb { x } _ { t } + \pmb { K } _ { p } \pmb { x } _ { t - 1 } + \pmb { K } _ { c } \pmb { c } _ { t }$ , where $\mathbf { } c _ { t }$ is a fixed oscillatory clock. These key and value projections incorporate causal 1D convolutions with a kernel size of two. The memory is read with the query $\pmb q _ { t } = \pmb { W } _ { q , x } \pmb x _ { t } + \pmb { W } _ { q , c } \pmb c _ { t }$ . The tracking vector is reflected along the axis $\pmb { u } _ { t } = \pmb { U } \pmb { x } _ { t } / \| \pmb { U } \pmb { x } _ { t } \|$ with strength $\beta _ { t } = 2 \sigma ( \pmb { w } _ { \beta } ^ { \top } \pmb { x } _ { t } + b _ { \beta } )$ . The maps $K _ { x } , K _ { p } , K _ { c } , V _ { x } , V _ { p } ,$ , and U are fixed random semi-orthogonal projections, so the only learned parameters are the query maps $( W _ { q , x } , W _ { q , c } ) _ { \cdot }$ the retention $\left( W _ { \lambda } , b _ { \lambda } \right)$ , and the strength $( { \pmb w } _ { \beta } , b _ { \beta } )$ . Neither state enters the other’s update, and $\boldsymbol { r } _ { t } \in \mathbb { R } ^ { d _ { v } + G }$ is passed to the output function $g .$ In practice, SRM uses $N _ { \mathrm { h } }$ read heads with separate query maps and concatenates their reads, so its output has width $N _ { \mathrm { h } } d _ { v } + G$

## 3.1 Fixed content–time write

Under one-step BPTT, $h _ { t - 1 }$ is treated as a constant, so feedback at time t has no gradient path to a write made at any earlier step. A learned write would therefore be trained only by the feedback available when it writes, which cannot indicate what a later step will need. SRM therefore fixes the write and learns the read, partly achieving D1. The key combines content, the current and previous input, with time, a clock whose rotors cycle at geometrically spaced frequencies (see Appendix C for more details), so a stored value can later be found by what preceded it or by when it occurred. The value carries both current and previous inputs, so a retrieval returns the pair. The fixed maps are semi-orthogonal, which keeps distinct inputs at distinct addresses, and the key is normalized, so every write has the same magnitude.

Key–value outer products are the classical linear associative memory (Willshaw et al. 1969, Anderson 1972, Kohonen 1972), revisited as fast weights (Schmidhuber 1992, Ba et al. 2016, Schlag et al. 2021) and linear attention (Katharopoulos et al. 2020). Fixing the input maps and learning only a readout is the idea of reservoir computing (Jaeger 2001, Maass et al. 2002), which SRM applies to the write alone. Keys that see the previous input are related to short convolutions in linear-recurrent layers (Fu et al. 2023, Kimi Team 2025) and to previous-token heads in Transformers (Olsson et al. 2022), and the clock follows sinusoidal and rotary position encodings (Vaswani et al. 2017, Su et al. 2024).

## 3.2 Learned associative read

For a memory $\textstyle \sum _ { i } k _ { i } v _ { i } ^ { \top }$ , the read $\begin{array} { r } { \pmb q _ { t } ^ { \top } \sum _ { i } \pmb { k } _ { i } \pmb { v } _ { i } ^ { \top } = \sum _ { i } \langle \pmb { q } _ { t } , \pmb { k } _ { i } \rangle \pmb { v } _ { i } } \end{array}$ returns the values whose keys match the query, plus interference from other keys that is small when keys are nearly orthogonal. The query has a content part and a time part, matching the key, so retrieval can select by content, by time, or both. We initialize $W _ { q , x }$ as a semi-orthogonal map scaled by $1 / \sqrt { d }$ and $W _ { q , c }$ at zero, so time addressing is learned. Because the query acts only on the current memory, its one-step BPTT gradient is exact (D1).

This is the read of linear attention and fast weights (Katharopoulos et al. 2020, Schlag et al. 2021) and of gated linear-attention and state-space models (Sun et al. 2023, Yang et al. 2024a, Dao & Gu 2024, Beck et al. 2024), whose capacity limits are well studied (Arora et al. 2024a;b, Okpekpe & Orvieto 2025). Delta-rule models reduce interference by learning to erase before writing (Schlag et al. 2021, Yang et al. 2024b; 2025). SRM relies on nearly orthogonal fixed keys instead, because learning what to erase would need gradient from later feedback.

## 3.3 Diagonal decaying retention

The retention should control how long memory persists without moving what is stored. Row s of the memory is scaled by $\lambda _ { t , s } = \exp ( - \exp ( { \pmb w } _ { \lambda , s } ^ { \top } { \pmb x } _ { t } + b _ { \lambda , s } ) ) \in ( 0 , 1 )$ , so every step is contractive (D4) and exact RTRL sensi tivities are elementwise and thus cheap to compute (D2). Unrolling gives $\begin{array} { r } { \pmb { H _ { t } } = \sum _ { \tau < t } \big ( \prod _ { s = \tau + 1 } ^ { t } \pmb { \Lambda } ( \pmb { x } _ { s } ) \big ) \pmb { k } _ { \tau } \pmb { v } _ { \tau } ^ { \top } } \end{array}$ a stored key keeps its coordinates up to a decay, so the query that matched it when it was written still matches it later. A transition that rotated or mixed key coordinates would move stored addresses in ways a query cannot undo (see Section 4.5). At initialization, $\pmb { w } _ { \lambda , s } = \mathbf { 0 }$ and $b _ { \lambda , s } = \log ( - \log r _ { s } )$ with $r _ { s }$ drawn from the LRU ring distribution (Orvieto et al. 2023), so the memory starts as leaky integrators over a range of time scales. Under one-step BPTT, the retention and the tracking strength are the only learned parameters whose gradient is used: they receive credit for their efect on the current step, but not for how their earlier values shaped the state being read.

Input-dependent diagonal gates, descending from LSTM and GRU gating (Hochreiter & Schmidhuber 1997, Cho et al. 2014), control retention in selective state-space, gated linear-recurrent, and gated linear-attention models (Gu & Dao 2024, De et al. 2024, Katsch 2023, Qin et al. 2024, Yang et al. 2024a, Feng et al. 2024); SRM’s double-exponential parameterization follows LRU (Orvieto et al. 2023). Such transitions commute, which limits order-sensitive computation (Merrill et al. 2024).

## 3.4 Householder tracking channel

Because the retention commutes, the memory cannot follow a state that evolves under non-commuting operations, such as which cup hides an object after a sequence of swaps. The tracking vector does this with the Householder transition (D3), which never expands ${ \boldsymbol { z } } _ { t }$ because $\beta _ { t } \in \left( 0 , 2 \right) \left( \mathrm { D } 4 \right)$ . The axis map U is fixed at a random semi-orthogonal initialization. The strength is initialized such that $\beta _ { t } = 1 . 9 \ ( w _ { \beta } = \mathbf { 0 }$ $b _ { \beta } = 2 \mathrm { a r c t a n h } ( 0 . 9 ) ,$ ).

DeltaNet and its gated variants (Schlag et al. 2021, Yang et al. 2024b; 2025, Kimi Team 2025) apply this transformation to the memory itself, with the written key as axis. SRM applies it to a separate small state with a fixed axis map, so tracking never moves the addresses of the associative memory.

## 4 Experiments

We first use a diagnostic task suite evaluate the units at initialization, the reliability of one-step BPTT gradients, and streaming learning performance. We then test SRM on Multi-Query Associative Recall (Arora et al. 2024a), streaming byte-level language modeling on TinyStories (Eldan & Li 2023), and control from delayed reward in an associative T-Maze, and finally ablate its components.

We choose four recurrent baselines for their varying diferences from SRM. Linear RTU (Elelimy et al. 2024) uses an input-independent complex-diagonal state transition. eLSTM (Irie et al. 2024) uses input-dependent gates and its recurrence is nonlinear in the state. GLA (Yang et al. 2024a), like SRM, is a linear attention unit with input-dependent diagonal gating, providing a streamlined architectural baseline. KDA (Kimi Team 2025) applies a generalized Householder update directly to the memory channel and shares SRM’s use of a 1D convolution. All learners operate under the streaming setting, that is, without replay, batches, or retained context. Performance is evaluated online: accuracy, success rate, or loss is measured on each sample before the learner updates on it. We present the key results here and give the full details in Appendix D.

## 4.1 Diagnostics

We introduce a task suite (Table 1 and Appendix D.3) developed to evaluate SRM along the design desiderata. The suite comprises ten tasks, each involving a diferent type of information, to provide a holistic evaluation of the ability to store and retain incoming information (D1). The suite covers tasks that test the capabilities to recall information by content or time, as well as to track counts, orders, and states. Two of the tasks (Orient and Shell) concern tracking non-commuting operations (D3).

<table><tr><td>Task</td><td>Target at query time</td></tr><tr><td>Presence</td><td>Whether a queried symbol appeared in the sequence.</td></tr><tr><td>Successor</td><td>The symbol that appeared right after the queried symbol.</td></tr><tr><td>Predecessor</td><td>The symbol that appeared right before the queried symbol.</td></tr><tr><td>Count</td><td>The number of occurrences of the queried symbol.</td></tr><tr><td>Order</td><td>Whether the first queried symbol appeared before the second in the presentation.</td></tr><tr><td>Position</td><td>The symbol presented at a queried position.</td></tr><tr><td>Parity</td><td>Addition modulo 2—tracking under the symmetric group  $S _ { 2 } .$ </td></tr><tr><td>Orient</td><td>Composition of rotations and reflections—tracking under the dihedral group  $D _ { 4 } .$ </td></tr><tr><td>Shell</td><td>Composition of cup permutations—tracking under the symmetric group  $S _ { 3 } .$ </td></tr><tr><td>Mod3</td><td>Addition modulo 3—accumulation in the cyclic group  $\mathbb { Z } _ { 3 }$ </td></tr></table>

Table 1: The task suite. Each task tests how well a unit encodes and retains one kind of information. Each sequence consists of 5 to 8 task-dependent input symbols, followed by a sequence of waiting symbols. At the end of the sequence, the learner is prompted to give an answer, at which point feedback is provided.

## 4.1.1 Memory Capabilities without Learning

We evaluate how well each unit can encode and retain information before any learning happens (D1). We unfold the units through a fixed set of examples, then continue the unfolding through a waiting period to test how well information is retained. We record the recurrent states after 0, 100, 200, and 500 of these waiting timesteps. At each checkpoint, we fit a fresh ridge probe on the states to predict the target (Figure 2); details in Appendix D.4.

SRM can encode and retain information well in most tasks. The previous-input projections, ${ K _ { p } } { \pmb x } _ { t - 1 }$ and $V _ { p } \mathbf { x } _ { t - 1 }$ , enable SRM to encode Successor and Predecessor well. The clock term, ${ K _ { c } } { c _ { t } }$ , is essential in Position. The probe falls short in Orient, and fails in Shell and Mod3 across all units, suggesting that the information is not present at initialization. Overall, compared to baseline units, SRM exhibits a greater capacity to store and retain information at initialization.

![](images/d2faeeb3ab4162e113c9bc3cf0bb59774a4587ae02af122ef6efcd6d853507a2.jpg)  
Figure 2: Accuracy of ridge-probe over recurrent states unfolded to delay checkpoints, without training. Lines connect points from each of the 10 random initializations. Dotted lines indicate chance-level accuracy.

## 4.1.2 Reliability of one-step BPTT in learning SRM parameters

We compare the reliability of learning with one-step BPTT relative to full BPTT. At time $T ,$ the latter computes the gradients for each parameter by accounting for the contributions of all steps in the past: $\mathbf { | } \mathbf { f u l l } =$ $\textstyle \sum _ { t = 0 } ^ { T } g _ { t }$ . On the other hand, one-step BPTT only uses the most recent step and omits the contributions from all earlier steps: $\scriptstyle { \pmb { g } } _ { \mathrm { c } }$ one\_ $\mathrm { \Pi } _ { \mathrm { s t e p } } = { \pmb g } _ { T } = { \pmb g } _ { \mathrm { f u l l } } - { \pmb g } _ { \mathrm { e a r l i e r \_ s t e p s } } .$ Learning with one-step BPTT becomes myopic unless the omitted $g _ { \mathrm { e a r l i e r \_ s t e p s } }$ is captured by $g _ { \mathrm { o n e \_ s t e p } } .$ We can measure the direction alignment between g<sub>earlier\_steps</sub> and ${ \pmb { g } } _ { \mathrm { o n e \_ s t e p } } ,$ for a parameter, with cosine similarity, and denote this alignment score as $\rho .$ Negative $\rho$ indicates that one-step BPTT degrades the parameters beyond one-step memory. The learning becomes unreliable if misalignment occurs frequently. We study how this alignment varies across tasks and changes as the loss evolves, for SRM.

For each task, we train the readout parameters to allow the loss to evolve in a controlled manner. While we keep all non-readout parameters fixed throughout this procedure, we diferentiate the loss with respect to selected parameters to obtain their alignment scores. We compute $\rho$ for the key projection matrix $\kappa$ (which denotes the concatenation of $K _ { x } , K _ { p } ,$ and $K _ { c } )$ , the value projection matrix V (concatenation of $V _ { x }$ and $V _ { p } )$ , and the gate’s parameters $W _ { \lambda } ,  { \boldsymbol } { b } _ { \lambda }$ . We additionally track $b _ { K }$ and $b _ { V }$ , two bias terms for the key and the value that are functionally fixed at 0 by SRM. Figure 3 shows a representative subset of results. We also compute the squared negative alignment, min $( 0 , \rho ) ^ { 2 }$ , and average it over all training steps and runs. Full details are in Appendix D.5.

![](images/0cfcbc0bc31bcec3202ed4c92bb5b025539f8c7a6b5129abd1f7a9b6806a7454.jpg)  
Figure 3: (Left) Alignment score (−1 to 1; higher is better) throughout readout training, for all of 10 SRM initializations. (Right) Squared negative alignment (0 to 1; lower is better), averaged over all steps and runs.

The results indicate that the learning of the gate parameters ${ \mathrm { i s } } ,$ at the very least, innocuous. Across all tasks and readout conditions, $\rho _ { W _ { \lambda } }$ rarely ever falls below 0. $\rho _ { \pmb { b } _ { \lambda } }$ stays consistently high, indicating that the retention strength is well learnable with one-step BPTT. For the write parameters, $\rho _ { K }$ and $\rho _ { V }$ remain near 0 throughout but occasionally become negative. Even though $\rho _ { b _ { K } }$ and $\rho _ { b _ { V } }$ are typically positive in Mod3, they exhibit high run-to-run variance in all tasks. In many runs, they sufer from negative alignment, sometimes severely, and do not improve with the condition of the readout. Overall, the results suggest that learning the write parameters with one-step BPTT is not a reliable process.

## 4.1.3 Memory Capabilities with Streaming Learning

We examine whether the observed nascent memory capabilities at initialization translate to realized capabilities, under one-step BPTT training (solid lines in Figure 4). On every task where untrained SRM has high probe accuracy relative to the other units, SRM trained under one-step BPTT also obtains an online accuracy higher than other units. This pattern is clear in all of the first seven tasks. In Orient, Shell and Mod3, where the ridge probe struggles, SRM performs well with training nevertheless. Overall, no unit outperforms SRM, across every task, when trained under one-step BPTT.

![](images/95fdf1b37da7fc6911f91350a43354c3e87a1908ccdd8e60d18362846b88283e.jpg)  
Figure 4: Training accuracy on the task suite under one-step BPTT (solid lines) and RTRL (dashed lines), averaged over 10 runs. Shading shows standard deviation for one-step BPTT. Dotted lines indicate chance.

We also report RTRL results (dashed lines in Figure 4). With one recurrent layer, RTRL updates are exact, and serve as an approximate upper bound performance on unit’s performance given its capacity. Compared to baseline units, SRM under one-step BPTT closely matches its RTRL performance, across all tasks, suggesting that its design is particularly efective for one-step BPTT, which also better satisfies (D2) than RTRL (Appendix E). From here on, we focus on one-step BPTT.

## 4.2 Streaming Learning on Multi-Query Associative Recall

We evaluate SRM on the Multi-Query Associative Recall (MQAR) task (Arora et al. 2024a), where the learner must recall multiple values associated with their corresponding keys. The demand to store multiple independent bindings simultaneously challenges associative memory with increased interference among bindings. Table 2 (and Figure 9a) reports the late-training accuracy on MQAR at three model dimensions across two task settings. Across tested model capacities, SRM outperforms all baselines, including KDA, which also uses causal 1D convolutions efective for this task. These results show that SRM’s gains extend beyond the task suite to established benchmarks for context-aware retention and associative binding.

## 4.3 Streaming Byte-Level Language Modeling

Next, we train language models one byte at a time on the TinyStories training set (Eldan & Li 2023). The learner reads the stories in dataset order with batch size one and resets its recurrent state at every story boundary. At each byte, it predicts the next byte, incurs the cross-entropy loss, and makes one update before the next byte arrives. Each model creates an embedding for it, passes it through four stacked residual blocks of recurrent units (see Appendix D.9), and then predicts the next byte with a linear head. All models are trained with one-step BPTT for 100M bytes, about 5% of the roughly 2B-byte corpus (see Figure 9).

Table 2(b) reports the online loss after 10M and 20M bytes, and Figure 9(b) in the Appendix shows the full learning curves. Among the baselines, RTU reaches the lowest online loss, followed by KDA and eLSTM, while GLA remains the highest throughout. SRM has the lowest online loss of all units from the first few million bytes onward and reaches around 0.82, below RTU, which achieves about 0.89. At this point, SRM can already produce locally coherent TinyStories-like text: its samples follow the typical story structure and are mostly grammatical within a sentence, although characters, pronouns, and events are not kept consistent across sentences (see Appendix D.9.3 for a generation sample). We also found that, in this regime, the optimizer matters as much as the unit: with RMSprop (Tieleman & Hinton 2012), the online loss can stop improving after a few million bytes and then drifts upward, whereas StreamingOptimizer (Elsayed et al. 2026) keeps decreasing (see Appendix D.9.1). Furthermore, we study performance scaling with parameter count in Appendix D.9.2, showing it scales under streaming learning.

<table><tr><td></td><td colspan="6">MQAR accuracy (%) ↑</td><td colspan="2">TinyStories loss ↓</td></tr><tr><td></td><td colspan="3">Sequence length 64</td><td colspan="3">Sequence length 128</td><td colspan="2">Bytes seen</td></tr><tr><td>Unit</td><td>Small</td><td>Medium</td><td>Large</td><td>Small</td><td>Medium</td><td>Large</td><td>10M</td><td>20M</td></tr><tr><td>GLA</td><td>11.1</td><td>17.6</td><td>23.3</td><td>1.3</td><td>8.3</td><td>11.3</td><td> $1 . 1 9 3 \pm 0 . 0 2 9$ </td><td> $1 . 1 9 2 \pm 0 . 0 4 8$ </td></tr><tr><td>KDA</td><td>5.6</td><td>20.1</td><td>92.4</td><td>2.7</td><td>5.2</td><td>62.3</td><td> $1 . 0 7 6 \pm 0 . 0 0 6$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 9$ </td></tr><tr><td>RTU</td><td>0.8</td><td>0.8</td><td>14.0</td><td>0.9</td><td>0.7</td><td>2.3</td><td> $0 . 9 4 2 \pm 0 . 0 0 2$ </td><td> $0 . 8 9 2 \pm 0 . 0 0 2$ </td></tr><tr><td>eLSTM</td><td>7.3</td><td>13.8</td><td>22.6</td><td>2.3</td><td>1.8</td><td>9.6</td><td> $1 . 3 0 7 \pm 0 . 0 1 7$ </td><td> $1 . 1 3 0 \pm 0 . 0 1 5$ </td></tr><tr><td>SRM</td><td>22.5</td><td>77.3</td><td>99.1</td><td>12.4</td><td>51.6</td><td>96.9</td><td> $\mathbf { 0 . 8 5 5 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 8 2 2 \pm 0 . 0 0 3 }$ </td></tr></table>

Table 2: MQAR: accuracy over the final 10% of training sequences at three FLOPs-matched model dimensions (Table 5), averaged over 10 runs. TinyStories: online next-byte loss averaged over the 1M bytes preceding the 10M- and 20M-byte marks, reported as mean ± standard deviation over three independent runs; all language models have around 13M parameters. Best value per column is shown in bold.

## 4.4 Memory-Dependent Control with Streaming Reinforcement Learning

Finally, we ask whether SRM’s binding and retrieval survive when the only feedback is a delayed scalar reward. In an associative variant of the T-Maze (Bakker 2001), the agent observes K key–value pairs one token at a time, each value being left or right, waits through a delay, and is then shown one of the keys at a junction; turning toward that key’s value gives a reward of +1 and the other turn −1. With K = 1 this is the standard memory T-Maze, and with $K > 1$ the agent must bind each value to its key, a reinforcementlearning counterpart of Successor. We used the Stream-AC algorithm (Elsayed et al. 2026) that updates at every step. We compare SRM against RTU (Elelimy et al. 2024) and report the average online success rate and the final success rate over ten independent runs (see Appendix D.8 for more details).

![](images/acde429b610191eb58510fe10613ae09f7e234a88629484ea678084e56e41e6a.jpg)  
Figure 5: Associative T-Maze with K key–value pairs. Left four panels: online success rate during training averaged over 10 runs with shaded area representing one standard error. Dotted line is chance. Rightmost panel shows online success over the final 10% of episodes (dots are individual seeds).

Figure 5 shows that both units solve $K = 1$ , but they have diferent performance once values must be bound to keys: SRM’s final online success stays above 0.99 for K = 2, 4, and 8, whereas RTU plateaus at 0.84, 0.67, and 0.63. We attribute the gap to the write rather than to credit assignment. SRM’s key includes the previous input, so each value is already stored under an address that identifies its key, and the reward only has to train the read.

## 4.5 Ablation studies

The ablation results in Figure 6 (Left) suggest that diferent tasks rely on diferent components of SRM, which together form a strong overall architecture. The previous-input projections $( K _ { p } \pmb { x } _ { t - 1 }$ and $V _ { p } { \pmb x } _ { t - 1 } )$ are essential in Predecessor (and also Successor). The clock is essential in Position, and also handy in Shell and Mod3. The tracking channel is necessary for SRM to perform non-commutative tracking reliably.

![](images/b4ebb0174f97a19d14572b3b1eb031f0951281fd29db9e2e880e4095421cbdc6.jpg)

![](images/14e0409f22602ab7b18bde8c46910216a470d7a5c5551bc8eb04440ef13716af.jpg)  
Figure 6: (Left) Ablation of SRM’s components: tracking channel, clock, and previous-input projections. Results show the average and standard deviation of 10 runs. $( \mathrm { R i g h t } )$ SRM’s recall accuracy degrades when the write weights (K and V) and their bias terms are made learnable, across various input alphabet sizes.

We study the role of freezing write parameters in SRM through a synthetic recall task in Figure 6 (Right). Training sequences are constructed using symbols randomly sampled from a discrete alphabet. The learner must recall the first-seen symbol at the end of the sequence. Enlarging the alphabet while keeping the model capacity fixed creates a bottleneck in the model’s ability to write distinct symbol representations into its recurrent state. The overlapping representation of distinct input symbols consequently leads to greater interference from other keys during retrieval.

Since learning write parameters with one-step BPTT is unreliable, SRM freezes them at a semi-orthogonal initialization. Compared with variants whose write weights (K and V) and bias terms are learnable, frozenwrite SRM achieves the highest recall accuracy and remains robust as the alphabet grows up to 512. We also overlay the learning curves of SRM variants with learnable write parameters in Figure 6 (Left) to showcase the role of freezing write parameters beyond the recall task.

Next, we investigate the design of SRM’s tracking channel. SRM applies its Householder transition to a separate vector ${ \boldsymbol { z } } _ { t }$ rather than to the memory $\pmb { H } _ { t }$ . We test this choice by merging the two states in two ways, with all parameters frozen at initialization so that the results reflect what the state update preserves rather than what training allows (Figure $7 ;$ details in Appendix D.12).

Reflecting the memory moves stored keys. Applying the reflection to the memory as well, with a constant strength $\beta ,$ gives $\pmb { H } _ { t } = ( \pmb { I } - \beta \pmb { u } _ { t } \pmb { u } _ { t } ^ { \top } ) \pmb { \Lambda } ( \pmb { x } _ { t } ) \pmb { H } _ { t - 1 } + \pmb { k } _ { t } \pmb { v } _ { t } ^ { \top }$ , as in delta-rule models (Yang et al. 2025, Kimi Team 2025). This update moves every stored key with each later input, so the query that matched it at write time no longer does. The measured overlap with the written position decreases with the number of subsequent inputs (Figure 7a). With $\beta = 1 . 9$ , retrieval after 32 and 64 inputs falls to 0.6 and 0.3, against 0.95 and 0.9 for SRM (Figure 7b).

Conversely, writing into the tracking vector corrupts it. In a five-cup shell game where each swap of cups i and $j$ reflects ${ \boldsymbol { z } } _ { t }$ along $e _ { i } - e _ { j }$ with $\beta = 2 , \ z _ { t }$ is exactly the one-hot position of the ball, and a linear probe is perfect at every length. We then add a write to this update, $\boldsymbol { z } _ { t } = ( \boldsymbol { I } - \beta \boldsymbol { u } _ { t } \boldsymbol { u } _ { t } ^ { \top } ) \boldsymbol { z } _ { t - 1 } + \eta \boldsymbol { w } _ { t }$ , where ${ \pmb w } _ { t } = { \pmb W } _ { w } { \pmb x } _ { t } / \| { \pmb W } _ { w } { \pmb x } _ { t } \|$ is the input under a fixed random map $W _ { w }$ , normalized to unit length, and η sets the size of the write relative to the unit-norm ${ \boldsymbol { z } } _ { t }$ . This lowers accuracy after 32 swaps to 0.6 for $\eta = 0 . 2 5$ and 0.3 for $\eta = 1 \ ( { \mathrm { F i g u r e } } \ { \mathrm { 7 c } } )$ . The memory alone is near chance after six swaps, because its diagonal transitions commute. SRM’s random-axis tracking vector is also near chance after 6 swaps, so its gains on state tracking (Figure 4) come from training.

## 5 Related Work

Streaming learning. Most deep learning methods assume access to the entire dataset, whereas real-world applications rely on a continuous stream of experience. A few supervised methods operate directly under the streaming setting (Hayes et al. 2019, Saran et al. 2023, Hayes & Kanan 2022), and related eforts have adapted reinforcement learning (Elsayed et al. 2026, Vasan et al. 2024), language modeling (Goyal et al. 2009), and video prediction (Carreira et al. 2024, Qian et al. 2024, Han et al. 2025) to the same regime. These works establish that streaming learning is feasible, but they largely consider fully observable settings where memory is not needed. Memory under streaming learning is what we study here.

![](images/9e0378d9812eadc54c3c3b13f2a686f50e564684e63ccf005a79354df63d4998.jpg)

![](images/0ed2d2c844ee671a7437624fa7ed7c51c224efdf8500360c9369d51e2f197fbb.jpg)

![](images/aa7bf01e49dda4dba3898cf53717479b92046aad1ce399b70b295b762890e934.jpg)  
Figure 7: Merging the tracking channel and the memory, with SRM at initialization. (a) Fraction of a stored key left at its written position after number of inputs. (b) Key–value retrieval accuracy; the dotted line is chance. (c) Linear-probe accuracy for the ball position in a five-cup shell game.

Recurrent and state-space sequence models. Linear recurrent and state-space models have re-emerged as eficient alternatives to attention, trading quadratic context for a bounded recurrent state (Orvieto et al. 2023, Gu & Dao 2024, De et al. 2024, Beck et al. 2024). Much of their eficiency comes from diagonal or complex-diagonal transitions and parallel-scan training (Katharopoulos et al. 2020, Sun et al. 2023, Katsch 2023, Qin et al. 2024, Feng et al. 2024). The same diagonal structure makes their transitions commute, which limits order-sensitive computation (Merrill et al. 2024), and these models are usually trained by backpropagation through time over windows with state resets at segment boundaries (Dai et al. 2019, Hutchins et al. 2022). SRM departs on both axes: it keeps a commuting diagonal retention for its associative memory but adds a separate non-commuting tracking channel, and it is designed to be trained in a streaming manner without resetting or storing contexts.

Online credit assignment. Real-time recurrent learning (RTRL) (Williams & Zipser 1989) computes exact gradients forward in time without storing the stream. Its cost has led to many approximations (e.g., Tallec & Ollivier 2018, Zucchet et al. 2023, Elelimy et al. 2024). SRM is designed for the streaming setting. Its write has no parameters and its tracking channel uses a fixed axis map. Exact RTRL is cheap for both components of SRM: the retention sensitivity is elementwise across memory rows, and the tracking-strength sensitivity has only G × (d + 1) entries.

## 6 Limitations and Conclusion

SRM’s fixed write bounds its capacity: stored values are separated only by near-orthogonal fixed keys, so interference grows with the number of items held, and SRM cannot learn what to erase. The tracking channel learns only its strength, so what it can track depends on the input representation. Our experiments are small in scale: diagnostic tasks with short sequences, a simple reinforcement-learning task, and small language models by current standards. We did not explore larger language models because of the compute cost of streaming learning: updating after every byte at batch size one limits parallelism across steps and makes it harder to exploit GPUs’ parallel processing capabilities. Hardware optimized for low-batch autoregressive decoding may help address this constraint, provided it also eficiently supports gradient updates.

We studied recurrent memory under streaming learning, with one-step BPTT, where a useful memory must keep past information retrievable while the feedback at each step can only shape how that state is used. Because this feedback cannot reach past writes, SRM fixes its content–time write and learns only how to read, and it places order-sensitive computation in a separate Householder tracking channel so that tracking never disturbs stored addresses. Across diagnostics, controlled tasks, and reinforcement learning, SRM learns memory-dependent behavior under one-step BPTT that the baselines often do not: it retains information that baselines lose, and in the associative T-Maze it binds and retrieves key–value pairs from delayed reward. It also reaches the lowest online loss among all units as a streaming byte-level language model. These results suggest that when credit cannot be carried back through time, how the state is written matters as much as how it is learned.

## References

James A. Anderson. A simple neural network generating an interactive memory. Mathematical Biosciences, 14:197–220, 1972.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Re. Zoology: Measuring and improving recall in eficient language models. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum? id=LY3ukUANko.

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, James Zou, Atri Rudra, and Christopher Re. Simple linear attention language models balance the recall-throughput tradeof. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2024b. URL https://openreview.net/forum?id=qRlcoPhEoD.

Jimmy Ba, Geofrey E. Hinton, Volodymyr Mnih, Joel Z. Leibo, and Catalin Ionescu. Using fast weights to attend to the recent past. In Advances in Neural Information Processing Systems, 2016.

Bram Bakker. Reinforcement learning with long short-term memory. In Advances in Neural Information Processing Systems, volume 14, 2001.

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michae Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long short-term memory. Advances in Neural Information Processing Systems, 37:107547–107603, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

João Carreira, Michael King, Viorica Patraucean, Dilara Gokay, Catalin Ionescu, Yi Yang, Daniel Zoran, Joseph Heyward, Carl Doersch, Yusuf Aytar, et al. Learning from one continuous video stream. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28751–28761, 2024.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 1724–1734, 2014.

Xiangxiang Dai, Yuejin Xie, Maoli Liu, Xuchuang Wang, Zhuohua Li, Huanyu Wang, and John Lui. Multi-agent conversational online learning for adaptive llm response identification. arXiv preprint arXiv:2501.01849, 2025.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pp. 2978–2988, 2019.

Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=ztn8FCR1td.

Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. Grifin: Mixing gated linear recurrences with local attention for eficient language models. arXiv preprint arXiv:2402.19427, 2024.

Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759, 2023.

Esraa Elelimy, Adam White, Michael Bowling, and Martha White. Real-time recurrent learning using trace units in reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=4UvMOnZMam.

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, and A Rupam Mahmood. Streaming deep reinforcement learning finally works. arXiv preprint arXiv:2410.14606, 2026.

Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, and Hossein Hajimirsadegh. Were rnns all we needed? arXiv preprint arXiv:2410.01201, 2024.

Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos: Towards language modeling with state space models. In International Conference on Learning Representations, 2023.

Amit Goyal, Hal Daumé III, and Suresh Venkatasubramanian. Streaming for large scale nlp: Language modeling. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, pp. 512–520, 2009.

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.

Riccardo Grazzi, Julien Siems, Arber Zela, Jörg Franke, Frank Hutter, et al. Unlocking state-tracking in linear rnns through negative eigenvalues. In International Conference on Learning Representations, volume 2025, pp. 36565–36597, 2025.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=tEYskw1VY2.

Albert Gu, Karan Goel, and Christopher Ré. Eficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022.

Tengda Han, Dilara Gokay, Joseph Heyward, Chuhan Zhang, Daniel Zoran, Viorica Pătrăucean, João Carreira, Dima Damen, and Andrew Zisserman. Learning from streaming video with orthogonal gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13651–13660, 2025.

Tyler L Hayes and Christopher Kanan. Online continual learning for embedded devices. In Conference on Lifelong Learning Agents, pp. 744–766. PMLR, 2022.

Tyler L Hayes, Nathan D Cahill, and Christopher Kanan. Memory eficient experience replay for streaming learning. In 2019 International Conference on Robotics and Automation (ICRA), pp. 9769–9776. IEEE, 2019.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Alston S. Householder. Unitary triangularization of a nonsymmetric matrix. Journal of the ACM, 5(4): 339–342, 1958.

DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent transformers. Advances in neural information processing systems, 35:33248–33261, 2022.

Kazuki Irie, Anand Gopalakrishnan, and Jürgen Schmidhuber. Exploring the promise and limits of real-time recurrent learning. In The Twelfth International Conference on Learning Representations, 2024.

Herbert Jaeger. The “echo state” approach to analysing and training recurrent neural networks. Technical Report GMD Report 148, German National Research Center for Information Technology, 2001.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Tobias Katsch. Gateloop: Fully data-controlled linear recurrence for sequence modeling. arXiv preprint arXiv:2311.01927, 2023.

Kimi Team. Kimi Linear: An expressive, eficient attention architecture. arXiv preprint arXiv:2510.26692, 2025.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Teuvo Kohonen. Correlation matrix memories. IEEE Transactions on Computers, C-21(4):353–359, 1972.

Wolfgang Maass, Thomas Natschläger, and Henry Markram. Real-time computing without stable states: A new framework for neural computation based on perturbations. Neural Computation, 14(11):2531–2560, 2002.

William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models. In International Conference on Machine Learning, pp. 35492–35506. PMLR, 2024.

John Miller and Moritz Hardt. Stable recurrent models. In International Conference on Learning Representations, 2019.

Destiny Okpekpe and Antonio Orvieto. Revisiting associative recall in modern recurrent models. arXiv preprint arXiv:2508.19029, 2025.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning, pp. 26670–26698. PMLR, 2023.

Emilio Parisotto and Ruslan Salakhutdinov. Neural map: Structured memory for deep reinforcement learning. arXiv preprint arXiv:1702.08360, 2017.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the dificulty of training recurrent neural networks. In International conference on machine learning, pp. 1310–1318. Pmlr, 2013.

Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. Advances in Neural Information Processing Systems, 37:119336–119360, 2024.

Zhen Qin, Songlin Yang, Weixuan Sun, Xuyang Shen, Dong Li, Weigao Sun, and Yiran Zhong. HGRN2: Gated linear RNNs with state expansion. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=y6SqbJfCSk.

Akanksha Saran, Safoora Yousefi, Akshay Krishnamurthy, John Langford, and Jordan T Ash. Streaming active learning with deep neural networks. In International Conference on Machine Learning, pp. 30005– 30021. PMLR, 2023.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pp. 9355–9366. PMLR, 2021.

Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992.

Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. DeltaProduct: Improving state-tracking in linear RNNs via Householder products. In Advances in Neural Information Processing Systems, volume 38, 2025.

Jimmy T.H. Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=Ai8Hw3AXqks.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. Advances in neural information processing systems, 28, 2015.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Richard S Sutton. Learning to predict by the methods of temporal diferences. Machine learning, 3(1):9–44, 1988.

Corentin Tallec and Yann Ollivier. Unbiased online recurrent optimization. In International Conference on Learning Representations, 2018.

Tijmen Tieleman and Geofrey Hinton. Lecture 6.5—RMSprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Gautham Vasan, Mohamed Elsayed, Alireza Azimi, Jiamin He, Fahim Shariar, Colin Bellinger, Martha White, and A. Rupam Mahmood. Deep policy gradient methods without batch updates, target networks, or replay bufers. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 845–891. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 019ef89617d539b15ed610ce8d1b76e1-Paper-Conference.pdf.

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar.org/CorpusID:13756489.

Vandi Verma, Mark W Maimone, Daniel M Gaines, Raymond Francis, Tara A Estlin, Stephen R Kuhn, Gregg R Rabideau, Steve A Chien, Michael M McHenry, Evan J Graser, et al. Autonomous robotics is driving perseverance rover’s progress on mars. Science Robotics, 8(80):eadi3099, 2023.

Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In International Conference on Learning Representations, 2015.

Ronald J. Williams and Jing Peng. An eficient gradient-based algorithm for on-line training of recurrent network trajectories. Neural Computation, 2(4):490–501, 1990. doi: 10.1162/neco.1990.2.4.490.

Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989. doi: 10.1162/neco.1989.1.2.270.

David J. Willshaw, O. Peter Buneman, and H. Christopher Longuet-Higgins. Non-holographic associative memory. Nature, 222:960–962, 1969.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-eficient training. In International Conference on Machine Learning, 2024a.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id=y8Rm4VNRPH.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In International Conference on Learning Representations, 2025.

Jason Yoo, Yingchen He, Saeid Naderiparizi, Dylan Green, Gido M van de Ven, Geof Pleiss, and Frank Wood. Lifelong learning of video difusion models from a single video stream. arXiv preprint arXiv:2406.04814, 2024.

Nicolas Zucchet and Antonio Orvieto. Recurrent neural networks: vanishing and exploding gradients are not the end of the story. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=46Jr4sgTWa.

Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, and Joao Sacramento. Online learning of longrange dependencies. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=Wa1GGPqjUn.

## A The One-step BPTT Algorithm

Algorithm 1 The One-step BPTT Algorithm   
1: Require: Recurrent function $f _ { \theta }$ with parameters θ   
2: Require: Output function $g _ { \phi }$ with parameters $\phi$   
3: Require: Data stream $\mathcal { D } ,$ step size α   
4: Initialize: $\theta , \phi ,$ hidden state $\mathbf { h } _ { 0 }  \mathbf { 0 }$   
5: for $t = 1 \ldots \infty$ do   
6: $\delta _ { W } \gets \mathbf { 0 } , \forall W \in \pmb { \theta } , \quad \delta _ { V } \gets \mathbf { 0 } , \forall V \in \phi$   
7: $\pmb { x } _ { t } , \pmb { y } _ { t } , \pmb { \mathrm { r e s e t } } \gets \mathcal { D }$   
8: if reset then   
9: $\mathbf { \Sigma } _ { h _ { t - 1 } }  \mathbf { \Delta } \mathbf { 0 }$   
10: ${ \pmb h } _ { t } \gets f _ { \pmb \theta } ( { \pmb h } _ { t - 1 } , { \pmb x } _ { t } )$ ▷ Forward pass   
11: $\hat { \pmb y } _ { t } \gets g _ { \phi } ( { \pmb h } _ { t } , { \pmb x } _ { t } )$   
12: $\mathrm { C o m p u t e } \ : \mathcal { L } _ { t } = \mathcal { L } ( \hat { \pmb { y } } _ { t } , \pmb { y } _ { t } )$   
13: $\begin{array} { r } { \delta _ { V } \gets \left( \frac { \partial \hat { \mathbf { y } } _ { t } } { \partial V } \right) ^ { \top } \frac { \partial \mathcal { L } _ { t } } { \partial \hat { \mathbf { y } } _ { t } } . } \end{array}$ $V \in \phi$ ▷ Gradient for $\phi$   
14: $\begin{array} { r } { \delta _ { \pmb { h } }  ( \frac { \partial \hat { \pmb { y } } _ { t } } { \partial h _ { t } } ) ^ { \top } \underline { { \frac { \partial \mathcal { L } _ { t } } { \partial \hat { \pmb { y } } _ { t } } } } } \end{array}$   
15: $\begin{array} { r } { \delta _ { W } \gets \left( \frac { \partial h _ { t } } { \partial W } \right) ^ { \top } \delta _ { h } , \quad W \in \pmb { \theta } } \end{array}$ ▷ Gradient for θ   
16: $\pmb { W }  \pmb { W } - \alpha \delta \pmb { W } , \quad \pmb { W } \in \pmb { \theta }$ ▷ Parameter update for θ   
17: $V  V - \alpha \delta _ { V } , \quad V \in \phi$ ▷ Parameter update for ϕ

## B The RTRL Algorithm

Algorithm 2 The Real-time Recurrent Learning Algorithm   
1: Require: Recurrent function $f _ { \theta }$ with parameters θ   
2: Require: Output function $g _ { \phi }$ with parameters $\phi$   
3: Require: Data stream ${ \mathcal { D } } ,$ step size α   
4: Initialize: $\theta , \phi ,$ hidden state $\mathbf { h } _ { 0 }  \mathbf { 0 }$   
5: Initialize: Sensitivity matrix $S _ { 0 } ^ { W }  \mathbf { 0 } , \forall W \in \pmb { \theta }$ ▷ $\begin{array} { r } { S _ { 0 } ^ { W } = \frac { \mathrm { d } \mathbf { h } _ { 0 } } { \mathrm { d } W } } \end{array}$   
6: for $t = 1 \ldots \infty$ do   
7: $\delta _ { V } \gets \mathbf { 0 } , \forall V \in \phi , \delta _ { W } \gets \mathbf { 0 } , \forall W \in \theta$   
8: $\pmb { x } _ { t } , \pmb { y } _ { t } , \pmb { \mathrm { r e s e t } } \gets \mathcal { D }$   
9: if reset then   
10: $\mathbf { \Sigma } _ { h _ { t - 1 } } \gets \mathbf { \mathbf { 0 } }$   
11: h<sub>t</sub> $ f _ { \theta } ( \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } )$ ▷ Forward pass   
12: Update sensitivity: $\begin{array} { r } { S _ { t } ^ { W } \gets \frac { \partial \mathbf { h } _ { t } } { \partial W } + \frac { \partial \mathbf { h } _ { t } } { \partial \mathbf { h } _ { t - 1 } } S _ { t - 1 } ^ { W } } \end{array}$ $\forall W \in \theta$   
13: $\hat { \mathbf { y } } _ { t } \gets g _ { \phi } ( \mathbf { h } _ { t } , \mathbf { x } _ { t } )$   
14: Compute $\mathcal { L } _ { t } = \mathcal { L } ( \hat { \underline { { y } } } _ { t } , \pmb { y } _ { t } )$   
15: $\begin{array} { r } { \delta _ { V }  \delta _ { V } + ( \frac { \partial \hat { \mathbf { y } } _ { t } } { \partial V } ) ^ { \mathrm { ~ l ~ } } \frac { \partial \mathcal { L } _ { t } } { \partial \hat { \mathbf { y } } _ { t } } . } \end{array}$ $\forall V \in \phi$ ▷ Gradient for $\phi$   
16: $\begin{array} { r } { \delta _ { W }  \delta _ { W } + ( S _ { t } ^ { W } ) ^ { \top } ( \frac { \partial \hat { \mathbf { y } } _ { t } } { \partial \mathbf { h } _ { t } } ) ^ { \top } \frac { \partial \mathcal { L } _ { t } } { \partial \hat { \mathbf { y } } _ { t } } , \quad \forall W \in \pmb { \theta } } \end{array}$ ▷ Gradient for θ   
17: $\pmb { W }  \pmb { W } - \alpha \delta _ { \pmb { W } } , \quad \forall \pmb { W } \in \pmb { \theta }$ ▷ Parameter update for θ   
18: $V  V - \alpha \delta \mathbf { { \sigma } } , \quad \forall V \in \phi$ ▷ Parameter update for ϕ

## C The SRM Oscillatory Clock

SRM uses a parameter-free oscillatory clock to make time available to its associative memory. Let $\boldsymbol { c } _ { t } \in \mathbb { R } ^ { 2 C }$ contain $C$ two-dimensional rotors with frequencies $\omega = ( \omega _ { 1 } , \ldots , \omega _ { C } )$ . Each rotor follows the fixed rotation

$$
\pmb { R } ( \omega _ { i } ) = \left[ \frac { \cos \omega _ { i } } { \sin \omega _ { i } } \quad \begin{array} { c } { - \sin \omega _ { i } } \\ { \cos \omega _ { i } } \end{array} \right] \in \mathbb { R } ^ { 2 \times 2 } ,\tag{7}
$$

$$
\begin{array} { r } { \pmb { c } _ { t + 1 } ^ { ( i ) } = \pmb { R } ( \omega _ { i } ) \pmb { c } _ { t } ^ { ( i ) } , \qquad \pmb { c } _ { t } ^ { ( i ) } \in \mathbb { R } ^ { 2 } . } \end{array}\tag{8}
$$

The complete update is block diagonal,

$$
\pmb { R } ( \omega ) = \mathrm { d i a g } \big ( \pmb { R } ( \omega _ { 1 } ) , \dots , \pmb { R } ( \omega _ { C } ) \big ) \in \mathbb { R } ^ { 2 C \times 2 C } ,\tag{9}
$$

$$
\begin{array} { r } { \pmb { c } _ { t + 1 } = \pmb { R } ( \omega ) \pmb { c } _ { t } . } \end{array}\tag{10}
$$

At the beginning of a stream, every rotor starts at phase zero with norm $1 / \sqrt { C } , \pmb { c } _ { 0 } ^ { ( i ) } = ( 1 , 0 ) ^ { \top } / \sqrt { C }$ . Consequently,

$$
\boldsymbol { c } _ { t } ^ { ( i ) } = \frac { 1 } { \sqrt { C } } \left[ \cos ( t \omega _ { i } ) \right] , \qquad \| \boldsymbol { c } _ { t } \| = 1 .\tag{11}
$$

The scale $1 / \sqrt { C }$ matters because the clock and the content terms are added in the same address space: with unit rotors, the clock would enter the key at norm $\sqrt { C }$ while the content terms enter at $O ( 1 )$ , so time would dominate content in every write address. Normalizing the clock to unit norm puts the two terms on a comparable scale. The clock is reset with the recurrent state at stream boundaries. It has no trainable parameters, is independent of the input, and preserves each rotor’s norm. We choose periods geometrically from 2 to a maximum period $P _ { \mathrm { m a x } }$

$$
P _ { i } = 2 ^ { 1 + \frac { i } { C - 1 } ( \log _ { 2 } P _ { \operatorname* { m a x } } - 1 ) } ,
$$

$$
\omega _ { i } = \frac { 2 \pi } { P _ { i } } .\tag{12}
$$

Short periods distinguish nearby steps, while longer periods provide a coarser signal over a longer horizon. The band is a resolution choice rather than a horizon choice, and it should not simply be made as large as possible: a rotor whose period greatly exceeds the horizon being resolved barely rotates, so it contributes an almost constant ofset to every write address and drives distinct addresses toward collinearity, which degrades retrieval. Because the periods are geometrically spaced and mutually incommensurate, a short band already separates timesteps far beyond $P _ { \mathrm { m a x } }$ . We use $P _ { \mathrm { m a x } } = 6 4$ and $C = 3 2$ rotors throughout. The clock enters both sides of associative retrieval. A write at step s and a read at step t use the addresses

$$
\bar { \pmb { k } } _ { s } = \pmb { K } _ { x } \pmb { x } _ { s } + \pmb { K } _ { p } \pmb { x } _ { s - 1 } + \pmb { K } _ { c } \pmb { c } _ { s } ,
$$

$$
\begin{array} { r } { \pmb q _ { t } = \pmb { W } _ { q , x } \pmb x _ { t } + \pmb { W } _ { q , c } \pmb c _ { t } , } \end{array}\tag{13}
$$

with $k _ { s } = \bar { k } _ { s } / \| \bar { k } _ { s } \|$ . Here, $K _ { x } , K _ { p } ,$ , and $\pmb { K } _ { c }$ are fixed projections, whereas $W _ { q , x }$ and $W _ { q , c }$ are learned. The clock, therefore, supplies a time address from initialization; learning determines how the query combines that signal with the content terms when retrieving a stored value.

The rotation geometry also makes relative ofsets available. For each rotor,

$$
\begin{array} { r } { \left. c _ { t } ^ { ( i ) } , c _ { s } ^ { ( i ) } \right. = \frac { 1 } { C } \cos \big ( ( t - s ) \omega _ { i } \big ) . } \end{array}\tag{14}
$$

The distinction is clearest for a single rotor, for which we drop the scale $1 / \sqrt { C }$ to reduce clutter. A fixed query using the initial clock phase $\pmb { c } _ { 0 } = ( 1 , 0 ) ^ { \top }$ gives

$$
\begin{array} { r } { \pmb { c } _ { 0 } ^ { \top } \pmb { c } _ { s } = \cos ( s \omega ) , } \end{array}\tag{15}
$$

which identifies the write by its absolute position s relative to the stream reset. In contrast, a query using the current clock phase gives

$$
\begin{array} { r } { \pmb { c } _ { t } ^ { \top } \pmb { c } _ { s } = \cos \big ( ( t - s ) \omega \big ) , } \end{array}\tag{16}
$$

which identifies the write by its separation $t - s$ from the current read. Thus, $c _ { 0 }$ is an absolute time reference, whereas the moving $\mathbf { } c _ { t }$ is a relative time reference. The learned query maps can use either comparison or combine them with content.

## D Experimental Details

## D.1 Baselines

The task suite (Figures 2 and 4) compares SRM with GLA (Yang et al. 2024a), KDA (Kimi Team 2025), and RTU (Elelimy et al. 2024). The associative T-Maze compares SRM with RTU, and the language-modeling experiment compares SRM with GLA, KDA, RTU, and eLSTM (Irie et al. 2024).

The original designs of GLA and KDA use a low-rank parameterization of some weights. In our experiments, we parameterize all weights with full-rank to isolate away any efects and implications of the low-rank parameterization choice, as SRM uses full-rank parameterization everywhere. For KDA, we use a convolution kernel size of two rather than the original four, matching that of SRM. We initialize every units (including SRM, as discussed in Section 3.3) with the memory-preserving initialization strategy for the input-dependent gates—weights initialized to 0, and bias initialized to the LRU ring distribution.

In every experiment, we match the size of the baseline units to SRM by FLOPs per timestep. Nonlinear operations are included in this computation, using a consistent FLOPs convention.

## D.2 Diagnostics

Throughout the experiments in section 4.1, we keep the architecture minimal to minimize confounding expressivity augmented by non-recurrent layers. This design allows us to isolate the capabilities that are provided by the recurrent unit itself. We use a frozen input projection (initialized orthogonally) of width 32, followed by a layer of the investigated recurrent unit, followed by a parameter-free RMSNorm layer (necessary for stable gradient flow), followed by a two-layer MLP classifier with a hidden dimension of 32 and GELU activation.

Table 3 specifies the state size of SRM and baseline units used in this architecture. FLOPs matching includes the cost of the recurrent unit’s forward pass and one-step truncated-BPTT backward pass, but excludes the cost of external input projection, classifier, optimizer updates, and RTRL trace maintenance. BPTT and RTRL variants use identical model size.

Table 3: Size of the recurrent unit of SRM and baselines used throughout the diagnostics experiments. SRM uses 8 rotors for the clock and 8 tracking dimensions of the tracking channel.
<table><tr><td>Unit</td><td>Input width</td><td>State width</td><td>Read heads</td><td>tracking dimension</td></tr><tr><td>SRM</td><td>32</td><td>32</td><td>2</td><td>8</td></tr><tr><td>GLA</td><td>32</td><td>42</td><td>2</td><td>一</td></tr><tr><td>KDA</td><td>32</td><td>20</td><td>2</td><td>一</td></tr><tr><td>RTU</td><td>32</td><td>60</td><td>一</td><td>一</td></tr><tr><td>eLSTM</td><td>32</td><td>45</td><td>一</td><td>一</td></tr></table>

## D.3 Task Suite

We probe the memory capabilities of each recurrent unit using a suite of ten synthetic tasks. Each task specifies a presentation alphabet $\Sigma ,$ a distribution over presentation sequences, and a target to predict. The presentation length L is sampled uniformly from {5, 6, 7, 8}. An example has the form

$$
x _ { 1 } , \ldots , x _ { L } , \bullet , q _ { 1 } , \ldots , q _ { m } ,
$$

where • $\notin \Sigma$ marks the end of the presentation and the $q _ { i }$ are optional task queries. Supervision occurs at • when $m = 0$ and at the final query otherwise.

The tasks are defined as follows:

Presence The presentation contains distinct symbols. Given $q \in \Sigma ,$ , predict whether q appeared.

Successor The presentation contains distinct symbols. Given a presented symbol q other than the final symbol, predict the symbol immediately following it.

Predecessor The presentation contains distinct symbols. Given a presented symbol q other than the initia symbol, predict the symbol immediately preceding it.

Count Given $q \in \Sigma$ , predict how many times q appeared. The target count is sampled from {0, 1, 2, 3}, and no symbol appears more than three times.

Order The presentation contains distinct symbols. Given two presented symbols $q _ { 1 }$ and $q _ { 2 }$ , predict whether $q _ { 1 }$ appeared before $q _ { 2 }$

Position The presentation contains distinct symbols from $\Sigma = \{ 0 , \ldots , 7 \}$ . Let $p _ { i }$ refer to the position $i < L$ from the presentation. Given $p _ { i } ,$ predict the symbol presented at position i.

Parity Symbols are sampled independently from {0, 1}. Predict $\textstyle \sum _ { i = 1 } ^ { L } x _ { i }$ mod $2 .$

Orient Let rT denote a quarter (clockwise) rotation and rF a reflection (along the axis perpendicular to the heading). Beginning at the heading $h _ { 0 }$ , apply the presented rotations and reflections and predict the final heading in $\{ h _ { 0 } , h _ { 1 } , h _ { 2 } , h _ { 3 } \}$

Shell Let $s _ { 0 1 } , \ s _ { 0 2 } ,$ , and $s _ { 1 2 }$ denote swaps of the corresponding pairs of cups. Beginning with the object under cup $c _ { 0 } , \mathrm { a p p l y }$ the presented swaps and predict its final cup in $\{ c _ { 0 } , c _ { 1 } , c _ { 2 } \}$

Mod3 Symbols are sampled independently from {0, 1, 2}. Predict $\textstyle \sum _ { i = 1 } ^ { L } x _ { i }$ mod 3.

<table><tr><td>Task</td><td>|Σ|</td><td>Chance</td><td colspan="2">Examples</td></tr><tr><td></td><td></td><td></td><td>Sequence</td><td>Target</td></tr><tr><td>Presence</td><td>16</td><td>0.594</td><td> $\langle 1 5 , 9 , 4 , 0 , 7 , \bullet , 8 \rangle$   $\langle 1 0 , 4 , 8 , 5 , 1 5 , \bullet , 4 \rangle$ </td><td>0</td></tr><tr><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>Successor</td><td>8</td><td>0.190</td><td>(7, 1, 2, 6, 0, ●, 1)  $\langle 5 , 1 , 2 , 7 , 6 , 3 , 4 , 0 , \bullet , 2 \rangle$ </td><td>2 7</td></tr><tr><td></td><td></td><td></td><td> $\langle 7 , 1 , 2 , 6 , 0 , \bullet , 2 \rangle$ </td><td></td></tr><tr><td>Predecessor</td><td>8</td><td>0.190</td><td> $\langle 5 , 1 , 2 , 7 , 6 , 3 , 4 , 0 , \bullet , 7 \rangle$ </td><td>1 2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Count</td><td>4</td><td>0.250</td><td>(0,1, 0,1,3, ●, 3) (3, 0, 2, 1, 0, 1, 0, •, 0)</td><td>1 3</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Order</td><td>8</td><td>0.500</td><td>(7,1,2, 6, 0, ●, 2,0)</td><td>1</td></tr><tr><td></td><td></td><td></td><td>(7, 3, 4, 2, 1, ●, 3, 1)</td><td>1</td></tr><tr><td>Position</td><td>8</td><td>0.159</td><td> $\langle 7 , 1 , 2 , 6 , 0 , 9 , p _ { 2 } \rangle$ </td><td>2</td></tr><tr><td></td><td></td><td></td><td> $\langle 5 , 1 , 2 , 7 , 6 , 3 , 4 , 0 , \bullet , p _ { 4 } \rangle$ </td><td>6</td></tr><tr><td>Parity</td><td>2</td><td>0.500</td><td>(1, 1, 0, 1, 1, ●)</td><td>0</td></tr><tr><td></td><td></td><td></td><td> $\langle 1 , 1 , 1 , 1 , 0 , 0 , 1 , 0 , \bullet \rangle$ </td><td>1</td></tr><tr><td>Orient</td><td>2</td><td>0.250</td><td> $\langle \mathrm { r F } , \mathrm { r F } , \mathrm { r T } , \mathrm { r F } , \mathrm { r F } , \bullet \rangle$ </td><td> $h _ { 1 }$ </td></tr><tr><td></td><td></td><td></td><td> $\langle \mathrm { r } \mathrm { T } , \mathrm { r } \mathrm { F } , \mathrm { r } \mathrm { T } , \mathrm { r } \mathrm { T } , \mathrm { r } \mathrm { F } , \mathrm { r } \mathrm { T } , \bullet \rangle$ </td><td> $h _ { \mathrm { 0 } }$ </td></tr><tr><td>Shell</td><td>3</td><td>0.333</td><td> $\langle s _ { 0 1 } , s _ { 1 2 } , s _ { 0 1 } , s _ { 0 2 } , s _ { 0 1 } , \bullet \rangle$ </td><td> $c _ { 1 }$ </td></tr><tr><td></td><td></td><td></td><td> $\langle s _ { 0 1 } , s _ { 0 1 } , s _ { 1 2 } , s _ { 0 1 } , s _ { 1 2 } , s _ { 0 1 } , \bullet \rangle$ </td><td> $c _ { 2 }$ </td></tr><tr><td>Mod3</td><td>3</td><td></td><td> $\langle 0 , 2 , 0 , 1 , 0 , \bullet \rangle$ </td><td>0</td></tr><tr><td></td><td></td><td>0.333</td><td> $\langle 1 , 1 , 0 , 2 , 2 , 0 , 0 , 1 , \bullet \rangle$ </td><td>1</td></tr></table>

Table 4: The synthetic task suite. |Σ| denotes the presentation alphabet size. Chance is the expected accuracy of a task-appropriate guesser, with variable presentation length accounted for.

## D.4 Memory Capabilities without Learning

For each task, we construct a fixed bank of 480 distinct examples, with 120 examples at each presentation length $L \in \{ 5 , 6 , 7 , 8 \}$ . The examples are generated according to the task distributions described in $\mathrm { A p - }$ pendix D.3. All units and initialization seeds are evaluated on the same bank for each task. For Parity and Orient, the 480 examples exhaust the complete presentation space.

To measure retention, we insert n repetitions of a waiting symbol $\circ \not \in \Sigma$ between the presentation and the end-of-presentation symbol. Thus, an example at delay n has the form

$$
x _ { 1 } , \ldots , x _ { L } , \underbrace { \circ , \ldots , \circ } _ { n { \mathrm { ~ w a i t i n g ~ s t e p s } } } , \bullet , \underbrace { q _ { 1 } , \ldots , q _ { m } } _ { \mathrm { t a s k ~ q u e r i e s } } .
$$

We supply a dataset of 480 examples, one for each of $n \in \{ 0 , 1 0 0 , 2 0 0 , 5 0 0 \}$ . The recurrent state is reset before every example, and the same set of 480 examples unfolds the recurrent state. All recurrent roll-outs are performed in double precision. After the final symbol, the recurrent state is flattened into a probe feature vector. We denote its size to be d. Stacking these vectors produces, for each delay, a feature matrix

$$
X _ { n } \in \mathbb { R } ^ { 4 8 0 \times d } .
$$

For Parity and Orient, the 480 presentations comprise the complete population for lengths $L \in \{ 5 , 6 , 7 , 8 \}$ We therefore fit and evaluate the ridge probe on the complete population using the fixed regularization parameter $\kappa = 1 0 ^ { - 4 }$ . Because the number of presentations difers across lengths, we weight them accordingly to ensure each length contributes equally to the final accuracy. The final score measures linear decodability over the complete finite population, rather than out-of-sample generalization.

For tasks other than Parity and Orient, to ensure out-of-sample generalization, we evaluate linear ridge probes using length-stratified nested four-fold cross-validation. In detail, both the outer and inner folds are stratified by presentation length. For each outer fold, we hold out 25% of the examples for testing. We then perform four-fold inner cross-validation with the remaining 75%, for each candidate value of the ridge regularization parameter

$$
\kappa \in \left\{ 1 0 ^ { - 8 } , 1 0 ^ { - 7 } , 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 0 ^ { 0 } , 1 0 ^ { 1 } , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \right\} .
$$

We select the κ value with the highest mean score across the four inner validation folds.

With the selected $\kappa ,$ the probe is refitted on the complete outer training set and evaluated on the held-out outer test fold. Features are standardized using statistics computed only from this corresponding training set; the intercept is not regularized. Each example receives exactly one out-of-fold prediction. For each outer validation fold, the accuracy is computed separately at each presentation length and then averaged equally across the four lengths. We report the mean accuracy across the four outer validation folds.

We repeat the experiment for 10 diferent instantiations of each recurrent unit. We use model capacity as specified in Table 3. We instantiate every unit using each specific initialization scheme designed for long retention. That includes the ring initialization, with $r _ { \mathrm { m i n } } = 0 . 9 7$ and $r _ { \mathrm { m a x } } = 0 . 9 9 9 5$ . At the same time, we control for the expressivity of the unit by avoiding initializing weights to 0 (weights at 0 would potentially sterilize some capabilities), ensuring the experiment would accurately find a unit’s nascent capabilities.

## D.5 Reliability of one-step BPTT in learning SRM parameters

We introduce the alignment score $\rho$ in Section 4.1.2 as a measurement of one-step BPTT gradient reliability, relative to Full BPTT. We explain it more here. Under full BPTT, when a loss occurs at timestep $T ,$ , the error is propagated backward to produce a distinct gradient contribution at each timestep, $g _ { t }$ for each learnable parameter. Since the parameter is reused across time, it receives the same update based on accumulated gradients. Full BPTT computes the gradients for a particular parameter by taking the aggregate of the gradients accumulated over time: $\textstyle g _ { \mathrm { f u l l } } = \sum _ { t = 0 } ^ { T } g _ { t }$

One-step BPTT retains only the gradients contribution at the most recent step $g _ { \mathrm { o n e \_ s t e p } } = g _ { T }$ , thereby omitting all the gradient contributions of earlier steps, whose sum we denote by

$$
g _ { \mathrm { e a r l i e r \_ s t e p s } } = \sum _ { t = 0 } ^ { T - 1 } g _ { t } = g _ { \mathrm { f u l l } } - g _ { \mathrm { o n e \_ s t e p } } .
$$

While an optimizer can scale the size of a gradient update, it cannot recover the direction of the gradients that the gradient estimator omits. Thus, learning any dependency longer than one step is only reliable if the gradient estimator can reliably produce gradients that are aligned with $g _ { \mathrm { e a r l i e r \_ s t e p s } }$ directionally. We measure this alignment, for a particular parameter, with the Frobenius cosine similarity, and denote the score as $\rho \colon$

$$
\rho _ { \mathrm { p a r a m } } = \frac { \langle g _ { \mathrm { o n e \_ s t e p } } , g _ { \mathrm { e a r l i e r \_ s t e p s } } \rangle _ { F } } { \left\| g _ { \mathrm { o n e \_ s t e p } } \right\| _ { F } \left\| g _ { \mathrm { e a r l i e r \_ s t e p s } } \right\| _ { F } } ,
$$

where $\langle A , B \rangle _ { F }$ denotes the Frobenius inner product and $| | A | | _ { F }$ denotes the Frobenius norm. A positive value is a promising sign that one-step BPTT may perform credit assignment for this parameter beyond one timestep. A negative value indicates that this gradient will move the parameter against learning longerthan-one-step memory. Frequent negative alignment would suggest that learning with one-step BPTT is harmful and unreliable.

We examine how $\rho$ behaves with respect to the task structure and the condition of the query and readout, in Experiment 4.1.2, by training the readout and query parameters while keeping everything else frozen. Given an initialization of SRM, this protocol allows the loss to evolve in a controlled manner, while keeping the score comparable throughout training. The alignment score $\rho$ is obtained for selected parameters of SRM by autograd and cosine similarity computation. We report the score $\rho$ for selected parameters throughout this training procedure for selected tasks in Figure 3, and for the whole suite, here in Figure $\begin{array} { r } { 8 , } \end{array}$ for 10 initializations of SRM.

Cosine Similarity of One-Step Gradients vs. Earlier-Steps Gradients  
![](images/1e7aeeaa4cc85436a51b0edb26fe193e3f576dc4c31c841fc7fbab8260e4821f.jpg)  
Write params. K b<sub>K</sub> V b<sub>V</sub> Gate params. W b

![](images/1d3aac25f2d61a65eb6328ae5ddd2ed5c5546041deecce6c69bdbc1b8fcb24b3.jpg)  
Figure 8: (Top) Alignment score for parameters of SRM over the training horizon of the readout, across the task suite, with all of non-readout parameters fixed throughout this training. Results are smoothed with a 5000-point centered rolling average. (Bottom) Squared negative alignment, averaged over training steps and runs. This value ranges from 0 to 1, with higher values indicating that local updates oppose the gradient contributions from earlier steps.

We note that this measurement $\rho$ is more meaningful for our purpose than the cosine similarity between $g _ { \mathrm { o n e \_ s t e p } }$ and $g _ { \mathrm { f u l l } }$ . The reason is that $g _ { \mathrm { f u l l } }$ contains the contribution of the one-step gradients, in unknown amount, inflating the directional alignment between them. This inflated score would be misleading for analyzing how well one-step BPTT can perform credit assignment beyond one step.

The squared negative alignment is obtained with $\operatorname* { m i n } ( 0 , \rho ) ^ { 2 }$ . It penalizes negative alignment quadratically while treating zero or positive alignment as neutral. This metric is deliberately risk-sensitive, emphasizing severe negative alignment that could lead to detrimental optimization updates, while remaining tolerant of small deviations below zero. We obtain the value of the mean squared negative alignment by taking the average across all training steps and runs.

## D.6 Memory Capabilities with Streaming Learning

In this experiment, we use the ring initialization for all units, with $r _ { \mathrm { m i n } } ~ = ~ 0 . 9$ and $r _ { \operatorname* { m a x } } = 0 . 9 9 9$ . We use the StreamingOptimizer (Elsayed et al. 2026) with $\beta _ { 2 } ~ = ~ 0 . 9 9$ . We perform an iterative search for hyperparameters of StreamingOptimizer in this experiment. We search over $\beta _ { 2 } \in \{ 0 . 0 , 0 . 9 , 0 . 9 9 9 , 0 . 9 9 9 5 \}$ , at the learning rate of $1 \times 1 0 ^ { - 3 }$ , and find that $\beta _ { 2 } = 0 . 9 9$ is the best across the task suite and recurrent units.

For each combination of units and tasks, we sweep over 3 seeds to choose the better learning rate between $1 \times 1 0 ^ { - 3 }$ and $3 \times 1 0 ^ { - 3 }$ . With the chosen learning rate, we report results over 10 diferent seeds. This process is done separately for BPTT and RTRL.

For each training run, we aggregate per-sequence accuracy using non-overlapping bins of 100 sequences. We then average the binned accuracies across 10 independent runs and apply a 50-point moving average to obtain the learning curves in Figure 4.

## D.7 Streaming Learning on MQAR

In our experiment, we reduce the alphabet size of standard MQAR configuration (Arora et al. 2024a) from 8, 192 to 256. This is necessary to allow any streaming learner we test to produce meaningful results. We also reserve a symbol, instead of using random non-query symbols, for waiting timesteps. The two settings of MQAR we report in Table 2 and Figure $9 ( \mathrm { a } ) - 6 4$ and 128 sequence length—use 4 and 8 training pairs respectively, matching the standard setup. We use an architecture of 2 blocks (Figure 10), followed by a final RMSNorm and a non-linear, GeLU activated classifier readout.

We set the model dimension of the baseline units to approximately match three SRM capacity levels in FLOPs per timestep, as listed in Table 5. The model dimension values specify the width of all of the following layers: input projection, recurrent state, and the hidden layer of the classifier readout.

For SRM, the number of clock rotors and the dimensions of the tracking channel are both scaled to $\textstyle { \frac { 1 } { 4 } }$ the model dimension. Matching FLOPS includes counting all model layers in one forward pass, one-step truncated-BPTT backward pass, and one optimizer update, including estimated nonlinear-operation costs. The same configurations are used for both variants of the task.

Table 5: FLOPs-matched model dimensions for SRM and baselines, for MQAR.
<table><tr><td colspan="4">Model dimension</td></tr><tr><td>Model</td><td>Small</td><td>Medium</td><td>Large</td></tr><tr><td>SRM</td><td>32</td><td>64</td><td>128</td></tr><tr><td>GLA</td><td>40</td><td>76</td><td>156</td></tr><tr><td>KDA</td><td>36</td><td>72</td><td>144</td></tr><tr><td>RTU</td><td>42</td><td>90</td><td>184</td></tr><tr><td>eLSTM</td><td>37</td><td>77</td><td>157</td></tr></table>

In MQAR, a loss occurs at every query timestep, which can appear multiple times within a training sequence. We accumulate gradients at each query timestep and update parameters once at the end of the sequence, keeping parameters fixed within each sequence.

We use the StreamingOptimizer with $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ . This value is chosen from a grid search over $\beta _ { 2 } \in$ $\{ 0 . 9 , 0 . 9 9 9 , 0 . 9 9 9 5 \}$ , at the learning rate of $1 \times 1 0 ^ { - 3 }$ , on the MQAR variant with 64 sequence length, for all units at the specified small model-dimension variant. With $\beta _ { 2 } = 0 . 9 9 9$ , we sweep over 3 runs to find the best learning rate in $\{ 3 \times 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 3 } \}$ , for each combination of unit and size, on the MQAR variant of 64 length. Then, we train each unit over 50, 000 sequences at each task setting, and report the results in Table 2 and Figure $9 ( \mathrm { a } )$ as the average accuracy of the final 10% training sequences, averaged over 10 independent runs.

Figure 9 plots the full results summarized in Table 2: the MQAR accuracies across model dimensions and the complete TinyStories online-loss curves of Section 4.3.

![](images/02c3ca16f8baa92e502ec93eac433d3221a0a82c5291bf7f5a7ac58242273585.jpg)  
Figure 9: (a) MQAR: Average accuracy across three model dimensions, with dashed and solid lines denoting the shorter and longer task settings. (b) Online next-byte loss on TinyStories, over 100M training bytes. Each line is one of three independent runs. All models use networks that have around 13M parameters.

## D.8 Associative T-Maze

The observation is a 12-dimensional one-hot vector with 8 key slots, 2 value slots $( \mathtt { l e f t } , \mathtt { r i g h t } )$ , a blank slot, and a query flag; at the junction, the queried key’s slot and the query flag are both set. Each episode draws K distinct keys uniformly from the 8 symbols, an independent uniform value for each key, and one of the K pairs uniformly as the query, so an episode lasts $2 K + D + 1$ steps with $D = 1 0$ . There are two actions. Actions before the junction are ignored, and the junction action ends the episode with reward +1 if it matches the queried value and −1 otherwise; all other rewards are 0.

The policy and value functions use separate networks with identical structure: a linear input projection with bias, one streaming block (Figure 10) with model width $M = 6 4$ and state width $S \ = \ 6 4 .$ , a final RMSNorm, and a linear head. SRM’s tracking strength $( { \pmb w } _ { \beta } , b _ { \beta } )$ is trained with one-step BPTT, whose sensitivity is a $G \times ( d + 1 )$ matrix and cheap for $G = 1 6 ;$ the query maps and the retention receive the one-step gradient. RTU is also trained with one-step BPTT. Both units use the ring initialization with $r _ { \mathrm { m i n } } = 0 . 9 7$ and $r _ { \mathrm { m a x } } = 0 . 9 9 9 5$ . The memoryless baseline is the SRM agent with its recurrent state reset before every step. The final online success rates of SRM are 0.998, 0.996, and 0.997 for $K = 2 , 4$ , and $8 ;$ the learning curves in Figure 5 are averaged over blocks of 1,000 episodes.

Every step produces one update of each network from the TD error $\delta _ { t } = r _ { t + 1 } + \gamma \hat { v } ( s _ { t + 1 } ) - \hat { v } ( s _ { t } )$ , with $\hat { v } ( s _ { t + 1 } ) = 0$ at the junction. Each parameter keeps an accumulating eligibility trace $e  \gamma \lambda _ { \mathrm { t r } } e + \nabla$ , reset at the end of each episode, and moves by α $\delta _ { t } e / ( \mu + \epsilon )$ , where $\mu$ is a running maximum of $| \delta _ { t } e |$ with decay 0.99995 and $\epsilon = 1 0 ^ { - 8 }$ . The actor’s gradient is that of log $\pi ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t } )$ plus the entropy bonus multiplied by $\mathrm { s i g n } ( \delta _ { t } )$ . Table 6 lists the settings.

For each unit and each $K ,$ we run the four combinations of step-size pair and entropy coeficient in Table 6 with three seeds and choose the combination with the highest mean cumulative online return. We then run the chosen combination on ten new seeds, which are the ones reported in Figure 5. The memoryless baseline uses SRM’s chosen combination. Because each episode’s return is ±1, the online success rate over any block of episodes is exactly $\textstyle { \frac { 1 } { 2 } } ( 1 + { \bar { R } } )$ , where $\bar { R }$ is the mean return over the block.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Discount  $\gamma ,$  trace decay  $\lambda _ { \mathrm { t r } }$ </td><td> $0 . 9 9 , \ 0 . 8$ </td></tr><tr><td>Step sizes (critic head α, recurrent networks  $\alpha _ { \mathrm { { r e c } } } )$ </td><td> $( 1 0 ^ { - 2 } , 1 0 ^ { - 3 } ) \mathrm { ~ o r ~ } ( 1 0 ^ { - 3 } , 1 0 ^ { - 2 } )$ </td></tr><tr><td>Actor head step size</td><td> $\dot { \alpha } / 2$ </td></tr><tr><td>Entropy coefficient</td><td> $_ { 0 . 1 }$ </td></tr><tr><td>Training episodes</td><td>20,000 (30,000 for  $K = 8 )$ </td></tr></table>

Table 6: Associative T-Maze settings. The step sizes and the entropy coeficient are selected per unit and per K.

## D.9 Streaming Language Modeling

The stream is the TinyStories training set (Eldan & Li 2023) read as raw bytes, so the vocabulary has 256 symbols. Stories are presented in dataset order, and the recurrent state is reset at each story boundary. Each model consists of a byte embedding, four blocks (Figure 10), and a linear output head. Every byte produces one StreamingOptimizer update with $\beta _ { 1 } = 0 , \beta _ { 2 } = 0 . 9 9 9 9 5 , \epsilon = 1 0 ^ { - 8 }$ , and step size $3 \times 1 0 ^ { - 4 } .$ , from the gradient of the next-byte cross-entropy computed with one-step BPTT. The online loss is logged as the mean over consecutive windows of 1,000 bytes, each byte being scored before the update it produces. Each unit is run with three seeds for 100M bytes. Table 7 lists the widths.

<table><tr><td>Unit</td><td>Model width M</td><td>State width S</td></tr><tr><td>RTU</td><td>1024</td><td>1024</td></tr><tr><td>eLSTM</td><td>1024</td><td>512</td></tr><tr><td>GLA</td><td>768</td><td>512</td></tr><tr><td>KDA</td><td>768</td><td>512</td></tr><tr><td>SRM</td><td>768</td><td>512</td></tr></table>

Table 7: Widths of the four-layer streaming language models. All models have approximately the same number of parameters around 13M.

![](images/68d22774c63cd6b75a458e37371a72a190090185a5b7bc4ec69957dbb9eefa4f.jpg)  
(a) Architecture

![](images/b3befb4b013e356e67cfe11d4274a9bf7a644de64797180e35b3818e005d5b45.jpg)  
(b) Gated RNN  
Figure 10: Architecture used throughout the paper. RN denotes RMSNorm. M is the model dimension and S is the state dimension. σ denotes a sigmoid nonlinearity.

In every streaming block, the gate and the output projection have Xavier-uniform weights, and the RMSNorm scales start at one. Recurrent units other than SRM use the initialization of their reference implementations. All units are trained with one-step BPTT: the recurrent state is carried from byte to byte, but the gradient of each update stops at the previous state.

SRM’s recurrent parameters are the exception to the above: they follow the scheme given in Section 3. Specifically, $K _ { x } , K _ { p } , K _ { c } , V _ { x } , V _ { p }$ , and U are semi-orthogonal; each head’s $W _ { q , x }$ is semi-orthogonal scaled by $1 / { \sqrt { d } }$ while $W _ { q , c }$ is zero; $W _ { \lambda } = \mathbf { 0 }$ with $b _ { \lambda , s } = \log ( - \log r _ { s } )$ from the ring draw; and the tracking strength starts at $\beta _ { t } = 1 . 9 \ ( w _ { \beta } = 0 , \ b _ { \beta } = 2 \mathrm { a r c t a n h } ( 0 . 9 ) )$ and is learned. We use $r _ { \mathrm { m i n } } = 0 . 0 , r _ { \mathrm { m a x } } = 1 . 0$ in the streaming language modeling experiments. The clock uses $P _ { \mathrm { m a x } } = 6 4$ with $C = 3 2$ rotors, the tracking channel has width $G = 1 6 \AA$ , and SRM uses $N _ { \mathrm { h } } = 4$ read heads with value width $d _ { v } \ : = \ : d / N _ { \mathrm { h } }$ so that the read has the same width as the block’s residual stream. In Figure 10, SRM’s input width d is the model dimension M, and its state dimension S is the number of memory rows.

## D.9.1 Optimizer Choice

Figure 11 compares RMSprop (Tieleman & Hinton 2012) across second-moment decay rates $\beta _ { 2 }$ with StreamingOptimizer (Elsayed et al. 2026), which is equivalent to Adamax (Kingma & Ba 2015) with $\beta _ { 1 } = 0$ for RTU trained with one-step BPTT on 30M bytes. With RMSprop, the online loss falls to about 1.2 within the first few million bytes and then rises to about 1.35 by 30M bytes for every $\beta _ { 2 }$ , whereas with StreamingOptimizer it keeps decreasing and ends near 0.88.

![](images/2aaa0b562908b5aa2aa8ea5e71c3ec43499113ac474c9f4f97a91407eb456490.jpg)  
Figure 11: Online next-byte loss of RTU trained with one-step BPTT under RMSprop with diferent $\beta _ { 2 }$ and under StreamingOptimizer (equivalent to Adamax with $\beta _ { 1 } = 0 )$ , over the first 30M bytes.

## D.9.2 Architectures of the Scaling Sweep

Tables 8 and 9 list the SRM and RTU configurations of the scaling sweep. The configurations of the two units are matched in parameter count, so the per-layer recurrent state they carry difers. Figure 12 shows the resulting next-byte loss: SRM reaches a lower loss than RTU at every size, both at matched parameters and at matched training FLOPs.

<table><tr><td>Params</td><td>Layers</td><td>Width M</td><td>Rows S</td><td>Memory  $S \times M / 4$ </td><td>State / layer</td><td>Total state</td></tr><tr><td>2.8M</td><td>3</td><td>384</td><td>256</td><td> $2 5 6 \times 9 6$ </td><td>25,040</td><td>75K</td></tr><tr><td>6.8M</td><td>4</td><td>512</td><td>384</td><td> $3 8 4 \times 1 2 8$ </td><td>49,744</td><td>199K</td></tr><tr><td>13.6M</td><td>4</td><td>768</td><td>512</td><td> $5 1 2 \times 1 9 2$ </td><td>99,152</td><td>397K</td></tr><tr><td>36.0M</td><td>6</td><td>1024</td><td>704</td><td> $7 0 4 \times 2 5 6$ </td><td>181,328</td><td>1.09M</td></tr><tr><td>71.5M</td><td>8</td><td>1280</td><td>832</td><td> $8 3 2 \times 3 2 0$ </td><td>267,600</td><td>2.14M</td></tr></table>

Table 8: SRM configurations of the scaling sweep. M is the model width and S is the number of memory rows.

![](images/acbec7a4eb485c019b6fb497fa2b9c52702d00e7ff72b9ac7103ea2f2a291381.jpg)

![](images/edffbb9c238ae94e3d5dc53709ab1188c8e698d6ff5c541f2210cd94cb64bf97.jpg)

Figure 12: Next-byte loss of SRM and RTU across the scaling sweep, against (a) the number of parameters and (b) the training FLOPs. Error bars show the standard error of the mean. The configurations are listed in Tables 8 and 9.
<table><tr><td>Params</td><td>Layers</td><td>Width M</td><td>State width S</td><td>State / layer</td><td>Total state</td></tr><tr><td>2.6M</td><td>3</td><td>512</td><td>512</td><td>512</td><td>1.5K</td></tr><tr><td>6.9M</td><td>4</td><td>736</td><td>736</td><td>736</td><td>2.9K</td></tr><tr><td>13.1M</td><td>4</td><td>1024</td><td>1024</td><td>1,024</td><td>4.1K</td></tr><tr><td>36.4M</td><td>6</td><td>1408</td><td>1408</td><td>1,408</td><td>8.4K</td></tr><tr><td>72.6M</td><td>8</td><td>1728</td><td>1728</td><td>1,728</td><td>13.8K</td></tr></table>

Table 9: RTU configurations of the scaling sweep. The state of width S is held as $S / 2$ complex units.

## D.9.3 Sample Story from Streaming SRM

The following story was generated byte by byte by the SRM language model of Section 4.3 after 100M training bytes, from the prompt “Once upon a time” with temperature 0.7 and top-k sampling with k = 50. The text is reproduced verbatim, including its errors.

Once upon a time, there was a little girl named Lily. She loved to play outside with his friends.   
One day, the park saw a big rock in her hand. Lily was so happy and felt very sad.

Benny was happy and proud of herself for little girl.

But then, a cool of colorful things are friends and they all wanted to help. They liked to play with his friends. One day, Tim found a big rock and the room was always happy. But then she got to school, but she always remember how to make it fit a new park in the box. They were both very sour feel sad. The doctor said that it was okay. He thought it was so funny. She would make some wood and she was so happy and proud. The end.

## D.10 Task Suite Ablation Study

No-tracking-channel ablates the tracking channel z away, leaving SRM with just the memory channel H. No-clock removes ${ K _ { c } } { c _ { t } }$ from the key projections and $W _ { q , c }$ from the query projections. No-previous-input projections remove ${ \cal K } _ { p } { \pmb x } _ { t - 1 }$ from the key projections and $V _ { p } { \pmb x } _ { t - 1 }$ from the value projections.

With components ablated away, we give back the capacity to components that are still present, by matching FLOPs per timestep to SRM. We use the same architecture, optimizer, hyperparameter values, initialization scheme, and learning rate tuning grid as in the task-suite streaming learning experiment described in Appendix D.6 and Appendix D.2.

## D.11 Frozen Write Parameters Ablation Study

K denotes the concatenation of the projection weights $K _ { x } , K _ { p } ,$ and $\pmb { K } _ { c }$ . V denotes the concatenation of the projection weights $V _ { x }$ and $V _ { p } .$

We test variants of SRM: with K learnable, with V learnable, with K and V both learnable, and with $\kappa$ V as well as their bias terms (that are functionally fixed at 0 by standard SRM) learnable. All variants use the same state size (no FLOPs matching here). We use the same architecture, optimizer, hyperparameter values, initialization scheme, and learning rate tuning grid as in the task-suite streaming learning experiment described in Appendix D.6 and Appendix D.2.

In the recall task, each sequence is 9 timesteps long, and the final query timestep shows a reserved query symbol. Symbols from the input alphabet are randomly sampled at every non-query timestep without replacement. At the final query-timestep, the learner is tasked with recalling the symbol appears at the first timestep of the sequence. A cross-entropy loss occurs at this final timestep. Figure 6 (Right) reports the evaluation accuracy at the end of training, on a held-out set of examples.

## D.12 Tracking-Channel Separation

The analysis of the SRM’s tracking channel in Section 4.5 uses the SRM implementation at initialization with input width $d = 6 4 , d _ { k } = 6 4$ memory rows, one read head, and value width $d _ { v } = 1 6$ . We set $r _ { \operatorname* { m i n } } = r _ { \operatorname* { m a x } } = 1$ so that the retention is the identity and the memory neither decays nor grows. No parameter is learned. Results are averaged over three independent runs.

Each sequence contains 4 key–value pairs, with keys drawn without replacement and values drawn uniformly from two vocabularies of 8 random unit vectors, followed by $\Delta$ inputs drawn uniformly from the unit sphere. The merged variant computes its axes as ${ \pmb u } _ { t } = { \pmb W } { \pmb x } _ { t } / \| { \pmb W } { \pmb x } _ { t } \|$ with a fixed random orthogonal $W \in \mathbb { R } ^ { d _ { k } \times d }$ so the axes are isotropic, and uses a constant $\beta .$ . For the last pair, whose key $\boldsymbol { k } _ { t _ { 0 } }$ is written at step $t _ { 0 } .$ the query is the written key $\begin{array} { r } { \mathbf { q } = k _ { t _ { 0 } } , } \end{array}$ which is also its position after $\Delta$ steps of SRM’s identity retention. Address survival is $q ^ { \top } { \tilde { k } } / \| q \| ^ { 2 }$ , where k<sup>˜</sup> is $k _ { t _ { 0 } }$ propagated alone through the same transitions. Retrieval reads $\pmb { q } ^ { \top } \pmb { H } _ { t } / \lVert \pmb { q } \rVert ^ { 2 }$ from the full memory and chooses the most cosine-similar of the 8 candidate values. Each point uses 2,000 sequences.

In the shell game, there are $G = 5$ cups and one token per transposition, each embedded as a random unit vector in $\mathbb { R } ^ { 6 4 } .$ ; the ball starts under the first cup and $z _ { \mathrm { 0 } } = e _ { \mathrm { 1 } }$ . For aligned axes, we set $U$ so that $\pmb { U } \pmb { x } = \pmb { e } _ { i } - \pmb { e } _ { j }$ for the token that swaps cups i and $j ,$ and set $\beta = 2$ . For random axes, U is SRM’s random semi-orthogonal projection and the strength $\beta _ { t } = 2 \sigma ( \pmb { w } _ { \beta } ^ { \top } \pmb { x } _ { t } + b _ { \beta } )$ keeps its default initial value, which is close to 1.9. The written variant adds ηw with ${ \pmb w } _ { t } = { \pmb W } _ { w } { \pmb x } _ { t } / \| { \pmb W } _ { w } { \pmb x } _ { t } \|$ for a fixed random $W _ { w }$ . The memory-only probe uses the flattened $\pmb { H } _ { t }$ (1,024 features). Each probe is a ridge regression with bias onto the one-hot ball position, fit to standardized features of 6,000 sequences split $6 0 / 2 0 / 2 0$ into training, validation, and test sets; the ridge coeficient is chosen from $\{ 1 0 ^ { - 4 } , \ldots , 1 0 ^ { 3 } \}$ on the validation set, and the probe is refit on training and validation data and evaluated on the test set.

## E SRM with RTRL

Cost Analysis. We instantiate the architecture and its parameters identically for one-step BPTT and RTRL and measure the FLOPs of one training timestep at batch size one. We do not count the warm-up timestep that initializes recurrent sensitivities and optimizer state. A synthetic gradient at the output logits drives backward computation. The measurement counts all model layers, RTRL sensitivity maintenance, and one StreamingOptimizer update. The cost of loss computation, gradient clipping, initialization, and data movement is excluded. We count a multiply-add as two FLOPs and assign fixed formula-based costs to nonlinear operations. Table 10 reports the resulting equivalent FLOPs per timestep for SRM training, for all models used in experiments.

Between the two streaming-learning modes, SRM with one-step BPTT matches our design principle of cheap computation requirement (D2) much better.

<table><tr><td>Model</td><td>one-step BPTT</td><td>RTRL</td><td>Ratio</td></tr><tr><td>Task Suite Model</td><td>113,796</td><td>201,543</td><td>1.771×</td></tr><tr><td>MQAR small</td><td>629,715</td><td>720,729</td><td>1.145×</td></tr><tr><td>MQAR medium</td><td>1,965,923</td><td>2,656,873</td><td>1.351×</td></tr><tr><td>MQAR large</td><td>6,767,235</td><td>12,150,921</td><td>1.796×</td></tr><tr><td>Language</td><td>259,341,436</td><td>1,773,230,984</td><td>6.837×</td></tr></table>

Table 10: FLOPs per timestep for SRM training with one-step BPTT and RTRL. Costs include forward computation, backward computation, and one optimizer update.

Gradients Analysis. In an architecture with only one recurrent layer, RTRL gradient is exact. However, in a multi-layers architecture, RTRL implementations usually track only within-layer forward-mode sensitivities and neglect cross-layer sensitivity terms. Consequently, the computation of the gradients that flow through layers is an approximation. We quantify this discrepancy by measuring, at each layer, the cosine similarity between the gradients computed by RTRL and the exact gradients computed by full BPTT. All computations are performed in double precision. Figure 13 shows the result for an architecture of four SRM layers, each preceded by RMSNorm with a detached normalization scale. Detaching the normalization scale preserves its forward stabilizing efect while excluding gradients through the scale from the gradient alignment comparison.

![](images/6757a9764af710cdfd89c1d9732bd59be1709600108d75f5cf1a0701e7810763.jpg)  
Figure 13: RTRL gradients alignment against exact full BPTT gradients throughout training.

For the top layer (Layer 4), the gradients estimated by RTRL match those computed by full BPTT throughout training. However, the approximation deviation received by the next layer is already significant. The gradients that the bottommost layer (Layer 1) receives, as estimated by RTRL, only has a cosine similarity of around 0.2, to the gradients computed by full BPTT.