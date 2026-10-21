# Model- and Layer-Level Closed-Loop Forecasting with SSM-Based Spiking Neural Networks

Erik Stolt

Sanja Karilanova

Ayc¸a Ozc¸elikkale<sup>¨</sup>

Department of Electrical Engineering

Department of Electrical Engineering

Uppsala University

Department of Electrical Engineering

Uppsala University

Uppsala, Sweden

Uppsala University

Uppsala, Sweden

erikstolt02@gmail.com

sanja.karilanova@angstrom.uu.se

Uppsala, Sweden

ayca.ozcelikkale@angstrom.uu.se

Abstract—Deep State-Space Models (Deep SSMs) have shown promising performance and computational benefits in processing of time-series data with complex temporal dependencies. Spiking neural networks (SNNs) constitute a biologically inspired class of deep SSMs, where computing blocks communicate through spikes, discrete single-bit events rather than continuous activation functions. A particularly important task in time-series processing is forecasting, i.e. predicting future values of a sequence from past observations alone. However, how to best perform forecasting with SNNs is an open question, where the standard autoregressive nature of the data processing with SNNs requires values that are not available during the forecasting phase. In this article, we address this challenge. We explore two closed-loop forecasting strategies, in which the network’s own generated values are fed back into the network as if they were observed past values: (i) model-level and (ii) layer-level closed-loop forecasting. We evaluate these approaches on the ETT-small datasets, a standard benchmark for time-series forecasting, under both spike-based and continuous activation functions. We compare the proposed closed-loop approaches against state-of-the-art methods such as SpaceTime, as well as the zero-masking baseline in S4. Both of the proposed approaches perform comparably to or better than zero-masking with S4, and in most cases match SpaceTime. These findings highlight the trade-off between continuous and low-bit quantized activations, and demonstrate that the proposed Deep SSM architectures provide a competitive framework for forecasting.

Index Terms—Neuromorphic, Forecasting, Spiking Neural Networks, State-Space Models

## I. INTRODUCTION

Accurate time-series modeling is central to signal processing and machine learning, with applications across healthcare, finance, and engineering [1]. Real-world sequences frequently exhibit seasonality, trends, and regime shifts, which makes efficient and accurate modeling challenging [1].

Deep State-Space Models (Deep SSMs) have recently emerged as effective architectures for long-sequence modeling. They cast sequence processing as a linear dynamical system and scale linearly with sequence length, O(T), in contrast to the quadratic $\mathcal { O } ( T ^ { 2 } )$ complexity of standard Transformer attention [2], [3]. Spiking neural networks (SNNs) can be viewed as a biologically inspired subclass of Deep SSMs whose units communicate through spikes, i.e. discrete single-bit events, rather than continuous activations. This makes them attractive for low-power and neuromorphic deployment, and recent work reports substantial energy savings over conventional models on time-series benchmarks [4].

A practical and demanding use of these models is forecasting: predicting the next F values, i.e. the forecast horizon, from the preceding P observations, i.e. the context horizon. Forecasting is difficult for autoregressive SSMs because the recursion that drives the hidden state requires input values that are simply not available beyond the context. A common workaround is to feed zeros in place of the missing inputs (zero-masking), but this discards information and tends to degrade rapidly as the horizon grows. An alternative is to close the loop, feeding the network’s own predictions back as surrogate inputs; how best to do this in a Deep SSM, and whether it remains effective under a spike-based activation, is not well understood.

This article investigates closed-loop forecasting with Deep SSMs on real-world data, and studies the effect of the activation function by comparing the continuous GELU activation with a spike-based Signed-Spiked (Signed-Spk.) activation. All models are evaluated with standard forecasting metrics on the ETT-small benchmark [5]. Our main contributions are as follows.

• We formulate and compare two closed-loop forecasting strategies for Deep SSMs: model-level closed-loop forecasting, in which the model’s output is fed back as the next input, and layer-level closed-loop forecasting, in which a dedicated decoder generates the next-step inputs through a trainable feedback matrix, following the design of SpaceTime [6].

• We provide a systematic, controlled evaluation of both strategies against a zero-masking baseline on the ETTsmall datasets, under identical training budgets and preprocessing.

• We study these strategies under both a continuous (GELU) and a spike-based (Signed-Spk.) activation, quantifying the accuracy–efficiency trade-off between continuous and low-bit activations.

## II. RELATED WORK

a) Deep state-space models.: Structured SSMs have become a strong tool for long-sequence modeling. S4 attains strong results on long-range dependency benchmarks using a structured parameterization of the state matrix [2]. [7] subsequently introduced S4D, which initializes the state matrix A as diagonal; S4D matches S4 in many settings while being simpler to implement and interpret, and we adopt it as our backbone.

b) Forecasting with Deep SSMs.: When S4 was introduced, it was evaluated on time-series forecasting by zeromasking the unknown portion of the input sequence [2]. [6] later proposed SPACETIME (ST), an encoder–decoder architecture tailored to forecasting that combines companionstructured SSMs, residual connections, linear layers, and GELU activations. Crucially, ST introduces a closed-loop strategy in which a trainable matrix K synthesizes future inputs to advance the decoder past the context. Our layerlevel strategy is inspired by this mechanism, but we apply the feedback matrix to the non-activated neuron outputs rather than to the hidden state (see Section IV). More broadly, simple linear models have been shown to be surprisingly competitive with Transformers for forecasting [3], motivating the study of lightweight SSM-based alternatives.

c) Spiking neural networks for forecasting.: SNNs have been explored primarily for the energy efficiency of timeseries prediction [4], [8]–[10]. [8] applied a spiking model to short-term electrical load forecasting, and [9] introduced a temporal-segment spiking neuron for time series. [4] evaluated SNNs across diverse forecasting benchmarks and found them substantially more energy-efficient than traditional methods, highlighting their potential for this task. In contrast to these works, we study how the forecasting recursion itself should be closed for SSM-based spiking models, and directly compare a spike-based activation against a continuous one within a single controlled setup.

## III. PROBLEM FORMULATION

## A. Preliminaries

## B. Discrete-Time State-Space Models

A discrete-time SSM with linear state dynamics and nonlinear output function can be written as

$$
\pmb { x } [ t + 1 ] = \pmb { A } \pmb { x } [ t ] + \pmb { B } \pmb { u } [ t ] ,\tag{1a}
$$

$$
\pmb { y } [ t + 1 ] = f \bigg ( \Re \left( \pmb { C } \pmb { x } [ t + 1 ] \right) \bigg ) ,\tag{1b}
$$

where $A \in \mathbb { C } ^ { n \times n }$ is a diagonal state-transition matrix, $\pmb { B } \in \mathbb { \Sigma }$ $\mathbb { R } ^ { n \times n _ { \mathrm { i n } } }$ maps the input into the state, and $C \in \mathbb { R } ^ { n _ { \mathrm { o u t } } \times n }$ maps the state to the output. The input, state, and output vectors have dimensions $\mathbf { \Delta } \mathbf { u } [ t ] \in \mathbb { R } ^ { n _ { \mathrm { i n } } } , \mathbf { \Delta } \mathbf { x } [ t ] \in \mathbb { C } ^ { n }$ , and $\pmb { y } [ t ] \in \mathbb { R } ^ { n _ { \mathrm { o u t } } }$ respectively. Here $\Re ( \cdot )$ denotes the real part and $f ( \cdot )$ is an arbitrary activation function.

![](images/ddf75ef823011ab0f73f412cca9a39b659c92b3964da30952d83c2b59c7ca204.jpg)  
Fig. 1: Deep SSM architecture with an input layer consisting of m input features, followed by L SSM layers, and an output layer consisting of $q$ output features.

1) Deep State-Space Models: A Deep SSM is constructed by embedding SSM-based neurons into a neural-network architecture. The network consists of an input layer of m channels, an output layer of q channels, and L SSM hidden layers. These SSM hidden layers are simply referred to as SSM layers. Each SSM layer concatenates h SSM neurons of the form (1) into an array; in this article all layers use the same number of neurons h and the same state dimension n. Stacking L such layers gives the network its depth. Consecutive layers are connected by mixing (weight) matrices $W \in \mathbb { R } ^ { h n _ { \mathrm { o u t } } \times h n _ { \mathrm { i n } } }$ and batch-normalization layers [11]. Figure 1 illustrates the architecture.

## C. Forecasting

Let $\pmb { U } | t ] \in \mathbb { R } ^ { m }$ denote the discrete-time input signal with m features for $t = 0 , \ldots , T - 1$ , where T is the sequence length, and let $\pmb { Y } [ t ] \in \mathbb { R } ^ { q }$ denote the corresponding q-feature output signal. For a forecast reference time $t = \tau$ , the context sequence of length P and the target sequence of length $F$ are

$$
\pmb { \mathscr { u } } _ { i } = [ \pmb { U } [ \tau - P ] , \dots , \pmb { U } [ \tau - 1 ] ] \in \mathbb { R } ^ { P \times m } ,\tag{2a}
$$

$$
\pmb { y } _ { i } = [ \pmb { Y } [ \tau ] , \dotsc , \pmb { Y } [ \tau + F - 1 ] ] \in \mathbb { R } ^ { F \times q } ,\tag{2b}
$$

respectively. Here, i indexes the sample, with $i = \tau - t _ { 0 } + 1$ for a rolling forecasting scenario with a start time $t _ { 0 }$ . The forecasting task is to learn a function $\pmb { f } _ { \theta } : \mathbb { R } ^ { P \times m }  \mathbb { R } ^ { F \times q }$ parameterized by $\theta ,$ such that the prediction $\hat { \pmb y } _ { i } ~ = ~ \pmb f _ { \theta } ( \pmb u _ { i } )$ approximates the true future sequence $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { i } }$ . The parameters are learned by minimizing a loss L( $\mathcal { L } ( \cdot )$ between predicted and true targets over the N training samples:

$$
\hat { \boldsymbol { \theta } } = \underset { \boldsymbol { \theta } } { \arg \operatorname* { m i n } } \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } \big ( \mathbf { \boldsymbol { f } } _ { \boldsymbol { \theta } } ( \boldsymbol { \mathbf { u } } _ { i } ) , \boldsymbol { \mathbf { y } } _ { i } \big ) .\tag{3}
$$

Here “approximate” means that $f _ { \theta } ( { \boldsymbol { \mathbf { \mathit { u } } } } _ { i } ) \approx { \boldsymbol { \mathbf { \mathit { y } } } } _ { i }$ , with closeness measured by an evaluation metric that may or may not coincide with the training loss. In this article we focus on univariate forecasting, i.e. $m = q = 1$ , where future values are predicted from past observed values (the context). This setting defines the forecasting objective and serves as the basis for the modeling approaches that follow.

Remark 1. In the rest of the article, we refer to $\mathbf { \Delta } \mathbf { u } _ { i }$ and y<sub>i</sub> of Eq. (2) or dropping the sample index i, u and y as the input and output, respectively. Accordingly, u is written in terms of its elements and without reference $t o \textit { \textbf { ‰} }$ $[ u [ 0 ] , \ldots , u [ P - 1 ] ]$ , and similarly for y.

## D. Forecasting with Deep State-Space Models

Consider a single SSM neuron in the first hidden layer. At each time step the SSM receives an input u[t] and, through the hidden state x[t] and the parameters A, B, C, produces an output $y [ t ]$ , as in Eq. (1). Since the observed input $u [ t ]$ is available only for $t = 0 , \ldots , P - 1$ , the recursion cannot be advanced beyond the context without an assumption about the unseen future inputs $u [ t ] , t \geq P$ . Hence, a network of SSM neurons cannot provide a forecasting output. The forecasting strategies in the next section differ precisely in how they supply these missing inputs to the network.

## IV. METHODOLOGY

We consider three ways of supplying the missing future inputs to a Deep SSM: a zero-masking baseline, and two closed-loop strategies that feed the network’s own predictions back as inputs. We describe each in turn.

## A. Baseline: Zero-Masking

The simplest strategy expands the input vector and assumes zero-valued inputs for all future time steps, i.e. $u [ t ] = 0$ for $t \geq P$ :

$$
u _ { i , \mathrm { m a s k } } = u _ { i } \bigoplus 0 _ { F } \equiv [ u _ { i } [ 0 ] , \ldots , u _ { i } [ P - 1 ] , 0 , \ldots , 0 ] ^ { \top }\tag{4}
$$

where i indexes the sample, L is concatenation, P is the context window, $F$ the forecast horizon, and $_ { 0 _ { F } \mathrm { ~ a ~ } }$ length-F sequence of zeros. The masked input $u _ { i , \mathrm { m a s k } }$ has length $P +$ $F ,$ , so the corresponding output produced by the SSM also has length $P + F$ . During the future portion the hidden state evolves under the autonomous dynamics $\pmb { x } [ t + 1 ] = \pmb { A } \pmb { x } [ t ]$ since the input contribution vanishes. We refer to this strategy as zero-masking (ZM); it is the simplest baseline in this work and the strategy used by S4 [2].

a) Loss:: For ZM we minimize the mean-squared error (MSE) over the forecast horizon,

$$
\mathcal { L } ( \hat { y } , y ) \equiv \frac { 1 } { F } \sum _ { t = P } ^ { P + F - 1 } \left. \hat { y } [ t ] - y [ t ] \right. ^ { 2 } ,\tag{5}
$$

which trains the model to produce accurate predictions at the future time steps.

## B. Closed-Loop Forecasting

Autoregressive rollout generates future predictions by feeding each prediction back as the input at the next time step. This section explores two variants of this idea, which differ in where the feedback enters the network. In both cases forecasting proceeds in two phases. In the open-loop phase the context window of length P is propagated through the network; in the subsequent closed-loop phase, starting from the last context step, the network generates its own inputs.

a) Open-loop phase: The open-loop phase propagates the observed context sequence u through the network. For each $t = 0 , \ldots , P - 1$ , the hidden states of all SSM neurons are updated and the corresponding outputs y[t] in Eq. (1) are produced. Once the context has been consumed, the closedloop phase begins.

b) Closed-loop phase: In the closed-loop phase the model generates the input for the next time step. In model-level closed-loop forecasting this is simply the model’s prediction at the previous step. In layer-level closed-loop forecasting, the next-step inputs are instead produced by the decoder through an additional trainable matrix K. We detail both below.

## C. Model-Level Closed-Loop Forecasting

In the model-level variant, the Deep SSM first consumes the P-step context to update the hidden state of every SSM neuron. $\mathbf { A } \mathbf { t } t = P$ forecasting begins, and the network output at time t is fed back as the input at time t + 1. Future predictions are therefore computed as

$$
\begin{array} { r } { \hat { y } [ t + 1 ] = f _ { \theta } ( \hat { y } [ t ] ) , \quad t = P , . . . , P + F - 1 . } \end{array}\tag{6}
$$

This mechanism is entirely dependent on the accuracy of each prediction, since errors are fed back into the model and accumulate over the forecast horizon. The model-level variant is trained with the same loss as ZM, Eq. (5).

## D. Layer-Level Closed-Loop Forecasting

The layer-level variant is inspired by SPACETIME (ST) [6], which uses an encoder–decoder structure. We assign the first $L - 1$ SSM layers to the encoder and the last SSM layer to the decoder.

a) Encoder: The encoder follows the standard forward pass of a deep SSM. It consists of h SSM neurons in each of its $L - 1$ layers and processes the input sequence u of length $P$ using the recurrence in Eq. (1) for $t = 0 , \ldots , P - 1$ . Between consecutive encoder layers, the neuron outputs are transformed by a mixing matrix $\dot { \mathbfcal W } ^ { ( \ell ) }$ and bias $\pmb { b } ^ { ( \ell ) }$ , for $\ell \in { 1 , \ldots , L - 2 }$

b) Decoder: The decoder processes the encoded output of the j-th encoder neuron, denoted $\bar { u } _ { j } [ t ] ~ \in ~ \mathbb { R }$ for $j \in \{ 1 , \ldots , h \}$ . Unlike the encoder, the decoder appends a trainable matrix K that generates the next-step inputs from the current neuron outputs:

$$
\begin{array} { r } { \pmb { x } _ { j } [ t + 1 ] = \pmb { A } _ { j } \pmb { x } _ { j } [ t ] + \pmb { B } _ { j } \bar { \ b { u } } _ { j } [ t ] , } \end{array}
$$

$$
z _ { j } [ t + 1 ] = \Re ( C _ { j } \pmb { x } _ { j } [ t + 1 ] ) ,\tag{7a}
$$

$$
y _ { j } [ t + 1 ] = f ( z _ { j } [ t + 1 ] ) ,\tag{7b}
$$

(7c)

$$
\left[ \begin{array} { l } { \hat { \bar { u } } _ { 1 } [ t + 1 ] } \\ { \vdots } \\ { \hat { \bar { u } } _ { h } [ t + 1 ] } \end{array} \right] = K \left[ \begin{array} { l } { z _ { 1 } [ t + 1 ] } \\ { \vdots } \\ { z _ { h } [ t + 1 ] } \end{array} \right]\tag{7d}
$$

for $t = 0 , \ldots , P + F - 1$ , with $\hat { \bar { u } } _ { j } [ t ] \in \mathbb { R }$ and $\ b { K } \in \mathbb { R } ^ { h \times h }$ Here $\hat { \bar { u } } _ { j } [ t ]$ is the predicted next-step input to the j-th decoder neuron. The matrix K adds $h ^ { 2 }$ trainable parameters relative to the ZM and model-level models, i.e. one entry per pair of decoder neurons. See Appendix for the number of trainable parameters introduced by the rest of the parameters in the model.

Our decoder differs from the original ST design in Eq. (7d): whereas ST applies K to the hidden state of each neuron, we apply it to the vector of non-activated neuron outputs $z _ { j }$ . Since these non-activated outputs are the quantities that directly form the forecast, applying K at this point creates a more direct link between the current decoder predictions and the inputs used at the next forecasting step.

c) Loss.: The layer-level loss combines three MSE terms: one aligning the encoder outputs with the decoder-generated outputs, one supervising predictions during the open-loop phase, and one supervising predictions during the closed-loop phase:

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \hat { y } , y , \hat { u } , \bar { u } ) = \frac { 1 } { P - 1 } \sum _ { t = 1 } ^ { P - 1 } \left\| \hat { y } [ t ] - y [ t ] \right\| ^ { 2 } } \\ { \displaystyle \qquad + \frac { 1 } { P - 1 } \sum _ { t = 1 } ^ { P - 1 } \left\| \hat { \bar { u } } [ t ] - \bar { u } [ t ] \right\| ^ { 2 } } \\ { \displaystyle \qquad + \frac { 1 } { F } \sum _ { t = P } ^ { P + F - 1 } \left\| \hat { y } [ t ] - y [ t ] \right\| ^ { 2 } . } \end{array}\tag{8}
$$

This loss encourages accurate predictions during both the open- and closed-loop phases while also supervising the model-generated inputs.

Remark 2. The difference between model- and layer-level closed-loop forecasting lies in the feedback mechanism. In model-level closed-loop forecasting, predictions are directly fed back into the first layer of the model. In layer-level closedloop forecasting, however, the feedback mechanism is handled exclusively by the decoder by feeding back u<sup>ˆ</sup>¯ as inputs to the decoder in the next time step.

## V. EXPERIMENTS SET-UP

## A. Architecture

All models use a Deep SSM with three SSM layers of 64 neurons each and a state dimension of 128 per neuron, together with linear input and output projections. The state matrix A is initialized as S4D-Lin [7], B is initialized to all ones, and C is initialized randomly with unit standard deviation. In layer-level closed-loop forecasting, K is initialized using Kaiming uniform initialization. The trainable parameter in the models constitute: for each SSM neurons $A _ { j } ^ { l } , B _ { j } ^ { l } , C _ { j } ^ { l }$ and when present $K _ { j } ^ { l }$ for $j ~ = ~ 1 , . . . h$ neurons and $\mathbf { \Psi } _ { l } = \mathbf { \dot { 1 } } , . . . L$ hidden layers; the weights with bias $W ^ { l }$ for $l = 1 , . . . L + 1$ connecting each layer in the network; and the trainable weight and bias in the normalization layer. The same architecture and initialization are used for every model in this work, so that differences in performance can be attributed to the forecasting strategy and the activation function rather than to capacity. Please see Appendix for hyperparameters used and the number of trainable parameters overview. All models are trained on NVIDIA Tesla T4 GPU with 16GB RAM.

We compare two activation functions: the continuous GELU and the spike-based Signed-Spiked activation,

$$
\mathrm { S i g n e d - S p i k e d } ( x ) = \mathrm { s g n } ( x ) \mathbb { I } ( | x | > \vartheta ) ,\tag{9}
$$

where $\vartheta \in \mathbb { R }$ is the firing threshold set to $\vartheta = 1$ and I(·) the indicator function. The Signed-Spiked activation emits a signed single-bit event $( - 1 , 0 , 0 \mathrm { r } + 1 )$ , yielding low-bit interlayer communication. As this function is non-differentiable, we train it with a box-car surrogate gradient in the backward pass [12], [13], while in the forward pass we use the spiking function.

## B. Datasets

We evaluate on the univariate Small Electricity Transformer Temperature (ETT-small) benchmark [5], which comprises the datasets ETTm1, ETTm2, ETTh1, and ETTh2 (provided as CSV). We omit ETTm2 since it shares many characteristics with ETTm1 and due to its size. These datasets record long time series from two electricity transformers at two stations over roughly 20 months [5]. They are widely used benchmarks for long-range dependency (LRD) tasks because they combine long- and short-term periodic patterns with numerous irregular fluctuations. Each dataset is split 0.6/0.2/0.2 into train/validation/test partitions and normalized using the mean and variance calculated from the training dataset.

## C. Evaluation Metrics

The aim of the experiments is to compare diagonal SSM models under different activation functions and forecasting strategy, and to position them relative to these existing methods on the standardized ETT datasets. Hence for all experiments we report the MSE and mean absolute error (MAE) over the forecast horizon as defined in Section IV for each forecasting strategy. The values are reported as average over five random initializations of the model. Comparisons to the S4 and SPACETIME numbers are quote from [6].

## VI. RESULTS

## A. Forecasting Performance

We report three models, S4D-ZM, S4D-Model-CL, and S4D-Layer-CL, where the suffix denotes the forecasting strategy (zero-masking, model-level closed-loop, and layer-level closed-loop) applied to the same S4D backbone. We also include the S4 [2] and SPACETIME [6] numbers as external references. Table Ia reports the short-horizon setting $( F = 2 4 )$ and Table Ib the long-horizon setting $( F = 9 6 )$

a) SPACETIME as an upper reference.: Across every dataset and horizon, SPACETIME attains the lowest errors (e.g. MSE 0.026 on ETTm1 at $F = 9 6$ , Table Ib). This is expected: SPACETIME adds a multi-layer perceptron on top of the SSMs, uses a companion (rather than diagonal) state matrix, and has more than double the trainable parameters of our models [6]. We therefore treat it as a well-resourced upper reference and focus our analysis on the four parameter-comparable, diagonal-SSM models: S4D-ZM, S4D-Model-CL, S4D-Layer-CL, and S4.

TABLE I: Univariate time-series forecasting results on three ETT-small datasets. Results are averaged over five random initializations; subscripts denote the standard deviation. Results for S4 and SPACETIME are taken from [6]. Bold denote the best and underline the second-best model per row.  
(a) Context lengths P ∈ {48, 96, 720} and forecast horizon F = 24. S4 and SPACETIME results from Table 7 in [6].
<table><tr><td></td><td></td><td></td><td colspan="2">S4D-ZM</td><td colspan="2">S4D-Model-CL</td><td colspan="2">S4D-Layer-CL</td><td colspan="2">S4</td><td colspan="2">SPACETIME</td></tr><tr><td>Activation</td><td>Dataset</td><td>P</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">GELU</td><td>ETTh1</td><td>720</td><td> $0 . 0 8 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 2 2 { \scriptstyle \pm . 0 1 }$ </td><td> $0 . 0 8 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 2 3 { \scriptstyle \pm . 0 0 }$ </td><td> $\underline { { 0 . 0 6 } } _ { \pm . 0 0 }$ </td><td> $\underline { { 0 . 1 8 } } _ { \pm . 0 1 }$ </td><td>0.06</td><td>0.19</td><td>0.03</td><td>0.12</td></tr><tr><td>ETTh2</td><td>48</td><td> $0 . 1 2 _ { \pm . 0 0 }$ </td><td> $0 . 2 6 _ { \pm . 0 1 }$ </td><td> $0 . 1 2 _ { \pm . 0 0 }$ </td><td> $0 . 2 5 { \scriptstyle \pm . 0 1 }$ </td><td> $0 . 1 1 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 2 4 { \scriptstyle \pm . 0 0 }$ </td><td>0.10</td><td>0.23</td><td>0.06</td><td>0.19</td></tr><tr><td>ETTm1</td><td>96</td><td> $0 . 0 2 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 0 9 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 0 2 _ { \pm . 0 0 }$ </td><td> $0 . 0 9 { \scriptstyle \pm . 0 0 }$ </td><td> $\underline { { 0 . 0 2 } } \underline { { \pm . 0 0 } }$ </td><td> $\underline { { 0 . 0 9 } } \underline { { \pm . 0 0 } }$ </td><td>0.02</td><td>0.12</td><td>0.01</td><td>0.07</td></tr><tr><td rowspan="3">Signed-Spk.</td><td>ETTh1</td><td>720</td><td> $\underline { { 0 . 1 3 } } _ { \pm . 0 1 }$ </td><td> $\underline { { 0 . 2 8 } } _ { \pm . 0 1 }$ </td><td> $0 . 1 3 _ { \pm . 0 1 }$ </td><td> $0 . 2 9 _ { \pm . 0 1 }$ </td><td> $\mathbf { 0 . 1 1 _ { \pm . 0 1 } }$ </td><td> $\mathbf { 0 . 2 6 _ { \pm . 0 2 } }$ </td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>ETTh2</td><td>48</td><td> $0 . 1 3 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 2 7 { \scriptstyle \pm . 0 0 }$ </td><td> $\underline { { 0 . 1 3 } } \underline { { \pm . 0 0 } }$ </td><td> $\underline { { 0 . 2 7 } } \underline { { \pm . 0 0 } }$ </td><td> $\mathbf { 0 . 1 3 _ { \pm . 0 0 } }$ </td><td> $\mathbf { 0 . 2 7 } \pm . 0 0$ </td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>ETTm1</td><td>96</td><td> $0 . 0 2 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 1 1 { \scriptstyle \pm . 0 0 }$ </td><td> ${ \bf 0 . 0 2 _ { \pm . 0 0 } }$ </td><td> ${ \bf 0 . 1 0 { \scriptstyle \pm . 0 0 } }$ </td><td> $\underline { { 0 . 0 2 } } \underline { { \pm . 0 0 } }$ </td><td> $\underline { { 0 . 1 0 } } _ { \pm . 0 0 }$ </td><td></td><td>I</td><td>一</td><td>一</td></tr></table>

(b) Context length P = 720 and forecast horizon F = 96. S4 and SPACETIME results from Table 1 in [6].
<table><tr><td></td><td></td><td colspan="2">S4D-ZM</td><td colspan="2">S4D-Model-CL</td><td colspan="2">S4D-Layer-CL</td><td colspan="2">S4</td><td colspan="2">SPACETIME</td></tr><tr><td>Activation</td><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">GELU</td><td>ETTh1</td><td> $0 . 1 7 { \scriptstyle \pm . 0 2 }$ </td><td> $0 . 3 3 { \scriptstyle \pm . 0 2 }$ </td><td> $0 . 1 9 { \scriptstyle \pm . 0 5 }$ </td><td> $0 . 3 5 { \scriptstyle \pm . 0 5 }$ </td><td> $\underline { { 0 . 1 4 } } _ { \pm . 0 2 }$ </td><td> $\underline { { 0 . 3 0 } } \underline { { \pm } } . 0 2$ </td><td>0.32</td><td>0.49</td><td>0.05</td><td>0.18</td></tr><tr><td>ETTh2</td><td> $0 . 3 5 { \scriptstyle \pm . 0 4 }$ </td><td> $0 . 4 8 { \scriptstyle \pm . 0 3 }$ </td><td> $0 . 3 5 { \scriptstyle \pm . 0 3 }$ </td><td> $0 . 4 7 { \scriptstyle \pm . 0 3 }$ </td><td> $\underline { { 0 . 3 1 } } _ { \pm . 0 2 }$ </td><td> $\underline { { 0 . 4 4 } } _ { \pm . 0 2 }$ </td><td>0.38</td><td>0.50</td><td>0.12</td><td>0.27</td></tr><tr><td>ETTm1</td><td> $0 . 1 0 { \scriptstyle \pm . 0 2 }$ </td><td> $0 . 2 4 { \scriptstyle \pm . 0 2 }$ </td><td> $0 . 0 7 { \scriptstyle \pm . 0 1 }$ </td><td> $0 . 2 0 { \scriptstyle \pm . 0 1 }$ </td><td> $\underline { { 0 . 0 6 } } _ { \pm . 0 1 }$ </td><td> $\underline { { 0 . 1 8 } } _ { \pm . 0 2 }$ </td><td>0.65</td><td>0.73</td><td>0.03</td><td>0.12</td></tr><tr><td rowspan="3">Signed-Spk.</td><td>ETTh1</td><td> $\mathbf { 0 . 1 8 \mathrm { _ { \pm . 0 2 } } }$ </td><td> $\mathbf { 0 . 3 4 _ { \pm . 0 2 } }$ </td><td> $0 . 2 0 { \scriptstyle \pm . 0 2 }$ </td><td> $0 . 3 6 _ { \pm . 0 2 }$ </td><td> $\underline { { 0 . 2 0 } } \underline { { \pm . 0 3 } }$ </td><td> $\underline { { 0 . 3 5 } } \pm . 0 3$ </td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>ETTh2</td><td> $\underline { { 0 . 3 0 } } \underline { { \pm . 0 2 } }$ </td><td> $0 . 4 4 { \scriptstyle \pm . 0 2 }$ </td><td> $\mathbf { 0 . 3 0 } _ { \pm . 0 2 }$ </td><td> $\mathbf { 0 . 4 4 } _ { \pm . 0 2 }$ </td><td> $\underline { { 0 . 3 0 } } _ { \pm . 0 4 }$ </td><td> $\underline { { 0 . 4 4 } } _ { \pm . 0 3 }$ </td><td></td><td></td><td></td><td>一</td></tr><tr><td>ETTm1</td><td> $\underline { { 0 . 0 9 } } \underline { { \pm . 0 0 } }$ </td><td> $\underline { { 0 . 2 3 } } \underline { { \pm . 0 1 } }$ </td><td> $0 . 1 0 { \scriptstyle \pm . 0 0 }$ </td><td> $0 . 2 4 { \scriptstyle \pm . 0 1 }$ </td><td> $\mathbf { 0 . 0 8 _ { \pm . 0 1 } }$ </td><td> $\mathbf { 0 . 2 1 } _ { \pm . 0 2 }$ </td><td></td><td></td><td>一</td><td></td></tr></table>

b) Short horizon $( F = 2 4 ) .$ : Among our three models with GELU (Table Ia), S4D-Layer-CL is the strongest of our models, and it also beats S4 on two of three datasets: ETTh1 (MSE 0.057 vs. 0.061) and ETTm1 (MSE 0.016 vs. 0.024). On ETTh2, however, S4 is the best of the four (MSE 0.095), with S4D-Layer-CL closest among our models (MSE 0.112). The zero-masking approaches (S4D-ZM and S4) remain competitive at the short horizon. Comparing activations, GELU yields uniformly lower errors than Signed-Spk. at $F = 2 4$ (e.g. on ETTh1, S4D-Layer-CL drops from 0.108 to 0.057). Within the Signed-Spk. block, S4D-Layer-CL is best on ETTh1 and ETTh2, while S4D-Model-CL is marginally best on ETTm1 (MSE 0.021).

c) Long horizon $( F ~ = ~ 9 6 ) . .$ : The picture changes markedly at F = 96 (Table Ib). All of our three models with GELU outperform S4 by a large margin on every dataset. The gap is most striking on ETTm1, where S4 has MSE 0.651 while S4D-Layer-CL reaches 0.060, and is also clear on ETTh1 (0.316 vs. 0.144). As at $F = 2 4$ , S4D-Layer-CL is the best of our GELU models on all three datasets. For activations, GELU again leads on ETTh1 and ETTm1, but on ETTh2 the spike-based activation is actually stronger, with all three Signed-Spk. models (MSE ≈ 0.302–0.304) below their GELU counterparts (MSE ≥ 0.308). No single strategy dominates across the Signed-Spk. results at this horizon: S4D-ZM is best on ETTh1, S4D-Model-CL on ETTh2, and S4D-Layer-CL on ETTm1, indicating that the best choice depends on the dataset.

d) Qualitative inspection.: Figure 3 show forecasting examples for $P \ = \ 7 2 0 .$ , and $F ~ = ~ 9 6 $ using Signed-Spk. activation. The figure showing the same forecasting examples with GELU activation can be found in Appendix. We can see visually that in many cases the forecasting predictions follow the general pattern of the ground truth. The spike-based rollouts in Fig. 3 are visibly more quantized than the GELU rollouts, yet remain as aligned with the target.

![](images/5e61b947d20680ad2d234f27c8a3c8c5f3f2ed70743f50122edd8648ac4560bf.jpg)  
Fig. 2: Average elapsed time in seconds per epoch when training on the ETTh1 dataset. All models scale approximately linear with context length.

## B. Training Time vs. Context Horizon

We also study empirical training-time scaling with the context length (Fig. 2). All models scale approximately linearly with context length, in line with the O(T) complexity of the SSM recurrence. A small vertical offset separates the GELU and Signed-Spk. curves; this might be attributed to our custom implementation of the Signed-Spk. activation, whose forward pass and surrogate-gradient backward pass add overhead. However the Signed-Spk. activation is compatible with neuromorphic hardware which at inference can reduces the energy consumption significantly [14].

Example Predictions | Closed-Loop | Complex <sub>A</sub> | Signed Spike  
![](images/f6f0586b0ad4e26cb7b74b537b5ed63a805ba5771ee0b8325643083933cbed84.jpg)

![](images/42dcd2e7804532f75ad82f11bd2ba768dff256e746a39906eda0818300e0f0b6.jpg)

![](images/4fa56162ab83d68d050affcd09591f4040b9940439a19771852ef039207fa9df.jpg)  
(a) S4D-ZM on ETTh1.

![](images/bedc32ec78666f85d8162300cbddb762c072a818a5b1d50ae4a9f981a3d510f8.jpg)

Example Predictions | SpaceTime | Complex <sub>A</sub> | Signed Spike  
![](images/523c0431bfbf75b70cdb7e30b24ce67e8347c8f1d24c42aca1b0fdfcd282492e.jpg)

(b) S4D-Model-CL on ETTh1.  
![](images/738972270aff6e4a9fac6863af54d5db674db38558c8e5f507cd2d7e0d85d490.jpg)  
(c) S4D-Layer-CL on ETTh1.

![](images/df5756cdbc3f4fba61a4793fd6fd861154c9136d44a3174b42ada20e76282860.jpg)

![](images/e24d8d848f04875f7b15d6b5c5c294f995258342dab5b1a82aaf0c99e63dd52b.jpg)

![](images/05b89048bd3949301e6080866f0f4b1a504c438e7bae5f1cd28a3dd8fdbdef98.jpg)

(d) S4D-ZM on ETTh2.  
![](images/6020f84514bd25a68d3066a219cd8be029bbf1ce958c0402cc42715de56dc4ae.jpg)

![](images/68a78e3ec97418ab00c58288bd62bcd68ad1d31ca9e2cb31e22b613dd82a3cce.jpg)

(e) S4D-Model-CL on ETTh2.  
![](images/ba833155eaa70643c60ffd573b97582d2fbb0aff9d90ace585eb302a1650a1e1.jpg)

![](images/587ebecc6ae62411babc041e999c5e1ff10a57c6eac7d148f1ecb00a91ce78e0.jpg)

(f) S4D-Layer-CL on ETTh2.  
![](images/2922b73cbf7a4f27efe524afca7f3803a6ddc38ef675852480e37ae729b99824.jpg)

![](images/8124c897cde12f1406991e4c51bd63706bd113bfb10487a9d47f9f54e4c6a45b.jpg)

![](images/746cd9649e64cdee0c873f58737f8370e62fd97f69e74fc92d8e079d5051c733.jpg)  
(g) S4D-ZM on ETTm1.

![](images/bef71afceced68533f597ff227fdb8ca40b5c106d38e78be0fede2d2f58a3b76.jpg)  
(h) S4D-Model-CL on ETTm1.

![](images/70ccd1e0399ac1aceab7109aa545387962866521473e9a8fcb764b0b9f1b04c3.jpg)  
(i) S4D-Layer-CL on ETTm1.  
Fig. 3: Example forecasting samples for all strategies on the ETT-small datasets with context length P = 720, forecast horizon $F = 9 6$ , and with Signed-Spk. activation.

## VII. CONCLUSION

We studied how to forecast with SSM-based Deep SSMs by comparing three strategies for supplying the missing future inputs, zero-masking, model-level closed-loop, and layer-level closed-loop, on the same S4D backbone, and under both a continuous (GELU) and a spike-based (Signed-Spk.) activation. On the ETT-small benchmark, the proposed closed-loop models are competitive while scaling approximately linearly with context length. At the short horizon $( F = 2 4 )$ , the layerlevel model is generally the strongest of our models and beats S4 on two of three datasets, though zero-masking remains competitive there. At the long horizon $( F = 9 6 )$ , all three of our strategies improve substantially over S4, indicating that closing the loop matters more as the horizon grows. The spikebased activation trails the continuous one in most settings but is competitive at longer horizons and is the stronger choice on ETTh2, illustrating a favorable accuracy–efficiency tradeoff for low-bit activations. As expected, the more heavily parameterized SPACETIME remains the best-performing model overall.

TABLE II: Model parameters used in the univariate forecasting of ETT-small.
<table><tr><td>Description</td><td>Parameter Value</td></tr><tr><td>Input dimension per SSM neuron</td><td> $n _ { \mathrm { i n } } = 1$ </td></tr><tr><td>Output dimension per SSM neuron</td><td> $n _ { \mathrm { o u t } } = 1$ </td></tr><tr><td>Number of stacked SSM neurons</td><td> $h = 6 4$ </td></tr><tr><td>Number of SSM layers</td><td> $L = 3$ </td></tr><tr><td>State dimension per neuron</td><td> $n = 1 2 8$ </td></tr><tr><td>Dropout rate Batch size</td><td> $0 . 0$ </td></tr><tr><td>BatchNorm</td><td> $B = 6 4$ </td></tr><tr><td></td><td>False</td></tr><tr><td>SSM learning rate</td><td> $\mathrm { { l r } _ { S S M } = 1 0 ^ { - 2 } }$ </td></tr><tr><td>Other learning rate</td><td> $\mathrm { { l r } _ { O t h e r } = 1 0 ^ { - 2 } }$ </td></tr><tr><td>Lr scheduler</td><td>Cosine</td></tr><tr><td>Minimum lr</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>SSM weight decay rate</td><td> $\omega _ { \mathrm { S S M } } = 1 0 ^ { - 4 }$ </td></tr><tr><td>Other parameters weight decay rate Optimizer</td><td> $\omega _ { \mathrm { O t h e r } } = 1 0 ^ { - 5 }$ </td></tr><tr><td>Number of epochs</td><td>AdamW  $\mathrm { { E p o c h s } = 1 0 }$ </td></tr><tr><td>Seeds</td><td> $0 , 1 , 2 , 3 , 4$ </td></tr><tr><td>SSM output activation function</td><td> $f = \mathrm { G E L U } / \mathrm { S i g n e d } { \cdot } \mathrm { S p k } .$ </td></tr><tr><td>Early stopping</td><td>False</td></tr><tr><td>Loss</td><td>MSE</td></tr><tr><td>Context length</td><td> $P = \{ 4 8 , 9 6 , 7 2 0 \}$ </td></tr><tr><td>Forecast horizon</td><td> $F \in \{ 2 4 , 9 6 \}$ </td></tr><tr><td>dtmin</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td> $\mathrm { { d t } _ { \operatorname* { m a x } } }$ </td><td> $1 0 ^ { - 1 }$ </td></tr><tr><td>Target variable</td><td> $\mathrm { O T }$ </td></tr><tr><td></td><td></td></tr><tr><td>Train/val/test split</td><td> $0 . 6 / 0 . 2 / 0 . 2$ </td></tr></table>

a) Limitations and future work.: Our study is confined to univariate ETT-small forecasting and to two horizons, and our comparisons to S4 and SPACETIME inherit differences in training budget and implementation. Natural next steps are to extend the evaluation to multivariate and longer-horizon settings and additional benchmarks, and to estimate the energy consumption of the spike-based models on neuromorphic hardware.

## VIII. APPENDIX

Table II shows the hyperparameters used in the trained models for which we present the results in the main text. We use hyperparameters motivated by related works.

In Table III, $\ell = 0$ corresponds to the input layer and $\ell =$ $L + 1$ to the output layer.

The trainable parameters of the Deep SSM are $A , B , C , W ^ { ( \ell ) }$ , and $\pmb { b } ^ { ( \ell ) }$ for $\begin{array} { r c l } { \ell } & { = } & { 0 , \ldots , L } \end{array}$ . For $W ^ { ( \ell ) }$ and $\pmb { b } ^ { ( \ell ) }$ , the number of trainable parameters equals the number of entries in the corresponding matrix and vector. For A, B and $^ { C , }$ the number of independent trainable parameters is reduced. In particular, due to the conjugate symmetry discussed by [7], only half of the entries are independently parameterized.

TABLE III: Notation used in the model architecture and in network layer ℓ.
<table><tr><td>Parameter</td><td>Description</td></tr><tr><td> $m$ </td><td>Number of input features in the input layer</td></tr><tr><td> $q$ </td><td>Number of output features in the output layer</td></tr><tr><td> $n _ { \mathrm { i n } }$ </td><td>Input dimension per SSM neuron</td></tr><tr><td> $n _ { \mathrm { o u t } }$ </td><td>Output dimension per SSM neuron</td></tr><tr><td> $h$ </td><td>Number of parallel SSM neurons</td></tr><tr><td> $L$ </td><td>Number of SSM layers</td></tr><tr><td> $_ n$ </td><td>State dimension per neuron</td></tr><tr><td> $d _ { \mathrm { i n } } \equiv h n _ { \mathrm { i n } }$ </td><td>Total input dimension to an SSM layer</td></tr><tr><td> $d _ { \mathrm { o u t } } \equiv h n _ { \mathrm { o u t } }$ </td><td>Total output dimension of an SSM layer</td></tr><tr><td> $\ell$ </td><td>Layer index  $\ell \in \{ 0 , \ldots , L + 1 \}$ </td></tr><tr><td> $\pmb { W } ^ { ( \ell ) }$ </td><td>Mixing matrix between layer l and  $\ell + 1$ </td></tr><tr><td> $\pmb { b } ^ { ( \ell ) }$ </td><td>Bias vector between layer l and  $\ell + 1$ </td></tr></table>

TABLE IV: Number of trainable parameters for the components of the Deep SSM. $\mathbf { \boldsymbol { W } } ^ { ( \ell ) }$ denotes the mixing matrix between layer ℓ and $\ell + 1$ with the corresponding bias vector $\pmb { b } ^ { ( \ell ) }$
<table><tr><td>Parameter</td><td>Number of Trainable Parameters</td></tr><tr><td> $W ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } } }$ </td><td> $d _ { \mathrm { i n } } d _ { \mathrm { o u t } }$ </td></tr><tr><td> $\pmb { b } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ </td><td> $d _ { \mathrm { i n } }$ </td></tr><tr><td> $\pmb { A } \in \mathbb { R } ^ { n } \ ( \mathrm { d i a g } )$ </td><td> $h n / 2$ </td></tr><tr><td> $A \in \mathbb { C } ^ { n } \ ( \mathrm { d i a g } )$ </td><td> $h n$ </td></tr><tr><td> $B \in \mathbb { R } ^ { n \times n _ { \mathrm { i n } } }$ </td><td> $n _ { \mathrm { i n } } n / 2$ </td></tr><tr><td> $C \in \mathbb { R } ^ { n _ { \mathrm { o u t } } \times n }$ </td><td> $n _ { \mathrm { o u t } } n / 2$ </td></tr></table>

The number of trainable parameters is summarized in Table IV.

Figure 4 is showing example forecasting samples for all strategies on the ETT-small datasets with GELU activation function. This is complementary to Figure 2 in the main text, which shows the same samples using Signed-Spk. activation function.

## REFERENCES

[1] R. H. Shumway and D. S. Stoffer, Time Series Analysis and Its Applications, 3rd ed. Springer, 2011.

[2] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences with structured state spaces,” in International Conference on Learning Representations, 2022.

[3] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, 2023, pp. 11 121–11 128.

[4] C. Lv, Y. Wang, D. Han, X. Zheng, X. Huang, and D. Li, “Efficient and effective time-series forecasting with spiking neural networks,” in Proceedings of the 41st International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 235, 2024.

[5] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in The Thirty-Fifth AAAI Conference on Artificial Intelli-

Example Predictions | SpaceTime | Complex A | GELU  
![](images/1702ff495de47bb0d3dcc8607e87a03df8ec73ec5915ca69d14bb16a1ebb6dca.jpg)

![](images/97f66c5a97daec88753c6a5d9816f5e379d7fbe852565d3400e86696ad76cde1.jpg)

![](images/414bfe5746743388b47f47c70fd4985dd19a7dec6ca7b9bf7f40b340e2aea3a1.jpg)

![](images/129511a6e8ec865f3bfb1fa4d472783d41fadd8293a875ddcbe62a2b83e71963.jpg)

![](images/76c3488b6f3daa3adb66018deca4be4af1a9248c3a3b60f1ea4d10a390e4f1d4.jpg)

![](images/86e1a8f10b3b90f49d90e81f3eb637e62b70e6ca95d612d2a0ad48bd41a2027b.jpg)  
(c) S4D-Layer-CL on ETTh1.

(a) S4D-ZM on ETTh1.  
(b) S4D-Model-CL on ETTh1.  
![](images/4184bc5b9e9d0113a9816e2da68ebc2199945b459e95cd1d08ce17e47706279d.jpg)

![](images/788d2db456ed56958b039423e67096233120a12cd1bdd477a4ced1cd8476990e.jpg)

![](images/605dac10d033f078b3cc804e83250ad6a6afaf36942f483b4eb531cd447d8940.jpg)

![](images/3c79a4e4a7e12cb9b67a8159b901e5b6ae6b90141f84f81cab650252418ab38b.jpg)  
(d) S4D-ZM on ETTh2.

![](images/dddea12ebc2e0122673fba2e3327a2faf41933771e323f73d9c779a1fb1dd0a3.jpg)  
(e) S4D-Model-CL on ETTh2.

![](images/a81ac982cbff505ecb6d5ffbaa873235ebd3784f5cf75707e7ebe0df0a345a1d.jpg)  
(f) S4D-Layer-CL on ETTh2.

![](images/ef095ac499ee823cf3cab5e114e79547a6e02e76bf82b40944af3ca1b68b8887.jpg)

![](images/7029c602f687102bc355f858c2e5a20dff2a71f8b4d94d8b82a0f5f16fb06ae2.jpg)

![](images/d7cb0a2e791244ee76bcc4145e2d6393f5ee579f4c1dd25e09ee8b24bb732391.jpg)

![](images/b2d69a8b8b002a39d8149817b89c7395cc5b65e4022c8abefd944267a4cb5432.jpg)  
(g) S4D-ZM on ETTm1.

![](images/9ef8e8e2592bc210db0357a877891c0be3e21914a71c9182a68cd4f178b8d1dd.jpg)  
(h) S4D-Model-CL on ETTm1.

![](images/1cbbba95ed0d501fecf5265cc3776c6342819e2cf1eba6a7524677aa0d844ef9.jpg)  
(i) S4D-Layer-CL on ETTm1.

Fig. 4: Example forecasting samples for all strategies on the ETT-small datasets with context length P = 720, forecast horizon F = 96, and with GELU activation.

gence, AAAI 2021, Virtual Conference, vol. 35, no. 12. AAAI Press, 2021, pp. 11 106–11 115.

[6] M. Zhang, K. K. Saab, M. Poli, T. Dao, K. Goel, and C. Re, “Effectively´ modeling time series with simple discrete state spaces,” 2023.

[7] A. Gu, A. Gupta, K. Goel, and C. Re, “On the parameterization and´ initialization of diagonal state space models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022.

[8] S. Kulkarni, S. P. Simon, and K. Sundareswaran, “A spiking neural network (snn) forecast engine for short-term electrical load forecasting,” Applied Soft Computing, vol. 13, no. 8, pp. 3628–3635, 2013.

[9] F. Shibo, W. Feng, X. Gao, P. Zhao, and Z. Shen, “TS-LIF: A temporal segment spiking neuron network for time series forecasting,” in The Thirteenth International Conference on Learning Representations, 2025.

[10] W. Wu, D. Huo, and H. Chen, “Spikf: Spiking fourier network for efficient long-term prediction,” in Forty-second International Conference on Machine Learning, 2025.

[11] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proceedings of the 32nd International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 37, 2015, pp. 448–456.

[12] E. O. Neftci, H. Mostafa, and F. Zenke, “Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks,” IEEE Signal Processing Magazine, vol. 36, no. 6, pp. 51–63, 2019.

[13] A. Bittar and P. N. Garner, “A surrogate gradient spiking baseline for speech command recognition,” Frontiers in Neuroscience, vol. Volume

[14] M. Davies, A. Wild, G. Orchard, Y. Sandamirskaya, G. A. F. Guerra, P. Joshi, P. Plank, and S. R. Risbud, “Advancing neuromorphic computing with Loihi: A survey of results and outlook,” Proceedings of the IEEE, vol. 109, no. 5, 2021.

16 - 2022, 2022.