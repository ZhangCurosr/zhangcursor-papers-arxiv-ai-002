# X-RACE: XAI-assisted Recurrent neural network Attribution for Channel Estimation

Abdul Karim Gizzini<sup>∗</sup>, Yahia Medjahdi<sup>§</sup>

<sup>∗</sup>University of Paris-Est Creteil (UPEC), LISSI/TincNET, F-94400, Vitry-sur-Seine, France.´

<sup>§</sup> IMT Nord Europe, Institut Mines Tel´ ecom, Centre for Digital Systems, F-59653 Villeneuve d’Ascq, France.´ Email: abdul-karim.gizzini@u-pec.fr, yahia.medjahdi@imt-nord-europe.fr

Abstract—Deep learning models, notably Long Short-Term Memory (LSTM), have demonstrated promising performance in channel estimation for high-mobility vehicular environments. However, their black-box nature and architectural overhead limit trustworthiness and efficiency. Classical explainable AI (XAI) methods rely on costly iterative processes, offering only inputlevel filtering without addressing architectural fine-tuning. To overcome these limitations, this paper proposes the XAI-assisted Recurrent neural network Attribution for Channel Estimation (X-RACE) framework. X-RACE uses a low-complexity, one-shot dual-optimization strategy to simultaneously evaluate and prune irrelevant input subcarriers and internal hidden units. Furthermore, we propose novel temporal XAI metrics: Saturation Time, Importance Drift, and Relevance Contrast to characterize the LSTM’s learning dynamics and memory convergence. Extensive simulations demonstrate that X-RACE reduces inference complexity by at least 44.1% while improving or preserving Bit Error Rate (BER) performance, outperforming classical XAI schemes. Index Terms—AI, XAI, channel estimation, LSTM, input filtering, architecture fine-tuning.

## I. INTRODUCTION

Artificial intelligence (AI) serves as a foundational pillar for next-generation 6G networks [1], [2], where Deep Learning (DL) models are employed in the physical layer to support the high data rate and latency demands of mission-critical applications [3], [4]. Reliable receiver operation depends on accurate channel estimation, which is challenging in highmobility vehicular environments due to severely doublyselective channels. To bypass the performance degradation of classical estimation schemes, DL-based post-processing models are widely adopted [5]. While early implementations employ memoryless Feedforward Neural Networks (FNNs) [6], they ignore underlying temporal correlations. Conversely, Long Short-Term Memory (LSTM) networks model long-term temporal dependencies, making them well-suited for highmobility tracking [7]. LSTM addresses double selectivity by combining an input feature vector capturing spectral snapshots with internal recurrent states tracking Doppler impact over time. Despite their superior accuracy, standard LSTMs operate as black boxes, making it difficult to understand how their dense internal parameters actually learn time-varying channel dynamics. This lack of transparency undermines trustworthiness in safety-critical applications [8], [9]. Furthermore, standard LSTMs often use excessive internal hidden units, leading to unnecessary computational overhead. Classical eXplainable Artificial Intelligence (XAI) schemes, such as Local

Interpretable Model-agnostic Explanations (LIME) [10] and SHapley Additive exPlanations (SHAP) [11], offer input-level feature filtering but fail to prune the LSTM architecture. Additionally, these iterative methods induce prohibitively high computational complexity. Alternatively, the uniform input downsampling proposed in [7] reduces the architecture empirically rather than a context-aware mechanism adapted to varying channel conditions.

To address these challenges, and inspired by [6], this paper proposes the XAI-assisted Recurrent neural network Attribution for Channel Estimation (X-RACE) framework. X-RACE utilizes a low-complexity, one-shot dual-optimization strategy that simultaneously introduces dynamic perturbation noise into a pre-trained LSTM’s inputs and internal memory state. This joint attribution isolates specific relevant subcarriers and internal hidden units that contribute to channel tracking. Furthermore, to explicitly capture the recursive nature of the LSTM, X-RACE incorporates novel XAI-based temporal metrics to evaluate model convergence and relevance stability. The main contributions of this work are summarized as follows:

• Propose X-RACE, a dual-optimization input and architectural attribution framework specifically designed for recurrent channel estimators in doubly-selective channels.

• Introduce sliding-window XAI-based metrics to measure the learning dynamics and memory convergence of the employed LSTM model.

• Demonstrate via extensive simulations that X-RACE’s input-architecture optimization prunes irrelevant inputs and internal hidden units, outperforming classical XAI schemes by substantially reducing computational complexity while preserving Bit Error Rate (BER) performance.

The remainder of this paper is organized as follows: Section II presents the LSTM-based channel estimation. Section III details the proposed X-RACE framework. Section IV analyzes performance and computational complexity, and Section V concludes the paper.

## II. LSTM-BASED CHANNEL ESTIMATION

We consider an Orthogonal Frequency-Division Multiplexing (OFDM) system. $K _ { \mathrm { o n } } = K _ { p } + K _ { d }$ represents the active subcarriers. $K _ { p }$ and $K _ { d }$ refer to the allocated pilot and data subcarriers, respectively. The i-th received OFDM symbol $\tilde { y } _ { i } \in \mathbb { C } ^ { K _ { \mathrm { o n } } \times 1 }$ can be expressed as:

$$
\tilde { \pmb { y } } _ { i } [ k ] = \tilde { \pmb { h } } _ { i } [ k ] \tilde { \pmb { x } } _ { i } [ k ] + \tilde { \pmb { e } } _ { i } [ k ] + \tilde { \pmb { v } } _ { i } [ k ] , \quad k \in \mathcal { K } _ { \mathrm { o n } }\tag{1}
$$

where $\tilde { \textbf { \textit { x } } } _ { i } ~ \in ~ \mathbb { C } ^ { K _ { \mathrm { o n } } \times 1 }$ and $\tilde { { \mathbf { h } } } _ { i } ~ \in ~ \mathbb { C } ^ { K _ { \mathrm { o n } } \times 1 }$ denote the i-th transmitted OFDM symbol and its respective time variant frequency-domain channel response. Moreover, $\tilde { \mathbf { v } } _ { i } \in \mathbb { C } ^ { K _ { \mathrm { o n } } \times 1 }$ and $\tilde { e } _ { i } \in \mathbb { C } ^ { K _ { \mathrm { o n } } \times 1 }$ represent the additive white Gaussian noise (AWGN) and the Doppler-induced inter-carrier interference.

LSTM models effectively track doubly-selective channels across OFDM frames. This is achieved by employing interacting gates to dynamically control the flow of information at each time-step i, according to four sequential stages:

1) Long-Term Memory Gating: The forget gate selectively discards irrelevant historical components from the long-term memory, outputting a scaling vector via the sigmoid activation function σ(·), such that:

$$
\pmb { f } _ { i } = \sigma ( \pmb { W } _ { f } \hat { \pmb { h } } _ { i } ^ { \prime } + \pmb { U } _ { f } \pmb { s } _ { i - 1 } + \pmb { b } _ { f } ) .\tag{2}
$$

$\pmb { W } _ { f } \in \mathbb { R } ^ { S \times 2 K _ { \mathrm { o n } } } , \pmb { U } _ { f } \in \mathbb { R } ^ { S \times S }$ , and $\pmb { b } _ { f } \in \mathbb { R } ^ { S \times 1 }$ are the forget gate input weights, recurrent weights, and bias, respectively. S is the hidden state dimension, $\breve { \hat { h } } _ { i } ^ { \prime } \in \mathbb { R } ^ { 2 K _ { \mathrm { o n } } \times 1 }$ is the current estimated channel, and $\pmb { s } _ { i - 1 } \in \mathbb { R } ^ { S \times 1 }$ represents the previous short-term hidden state.

2) Current Input Gating and Candidate Generation: Concurrently, the input gate evaluates the current channel estimate to determine the update magnitude for each memory coordinate, while a candidate cell state ${ \tilde { c } } _ { i }$ models potential adjustments via a tanh activation, such that:

$$
\pmb { g } _ { i } = \sigma ( \pmb { W } _ { g } \hat { h } _ { i } ^ { \prime } + \pmb { U } _ { g } \pmb { s } _ { i - 1 } + \pmb { b } _ { g } ) ,\tag{3}
$$

$$
\tilde { \pmb { c } } _ { i } = \operatorname { t a n h } ( \pmb { W _ { c } } \hat { \pmb { h } } _ { i } ^ { \prime } + \pmb { U _ { c } } \pmb { s } _ { i - 1 } + \pmb { b } _ { c } ) ,\tag{4}
$$

where $W _ { g } , W _ { c } \in \mathbb { R } ^ { S \times 2 K _ { \mathrm { o n } } } , U _ { g } , U _ { c } \in \mathbb { R } ^ { S \times S }$ , and $b _ { g } , b _ { c } \in$ $\mathbb { R } ^ { S \times 1 }$ are the input weights, recurrent weights, and biases for the input and candidate gates, respectively.

3) Long-Term Cell State Synchronization: The long-term cell state $\pmb { c } _ { i } \in \mathbb { R } ^ { S \times 1 }$ is updated by blending past and present information, using an element-wise Hadamard product ⊙ to apply the forget and input scales:

$$
\begin{array} { r } { \pmb { c } _ { i } = \pmb { f } _ { i } \odot \pmb { c } _ { i - 1 } + \pmb { g } _ { i } \odot \tilde { \pmb { c } } _ { i } . } \end{array}\tag{5}
$$

4) Hidden State and Output Generation: Finally, the updated hidden state $\mathbf { \boldsymbol { s } } _ { i }$ is constructed to provide the instantaneous channel tracking output. The output gate filters the synchronized cell state memory, such that:

$$
\pmb { o } _ { i } = \sigma ( \pmb { W } _ { o } \hat { h } _ { i } ^ { \prime } + \pmb { U } _ { o } \pmb { s } _ { i - 1 } + \bar { \pmb { b } } _ { o } ) ,\tag{6}
$$

$$
\begin{array} { r } { \pmb { s } _ { i } = \pmb { o } _ { i } \odot \operatorname { t a n h } ( \pmb { c } _ { i } ) . } \end{array}\tag{7}
$$

Finally, an FNN layer is employed to extract the final LSTM-based channel estimate $\hat { h } _ { \mathrm { L S T M } _ { i } } ~ \in ~ \mathbb { R } ^ { 2 K _ { \mathrm { o n } } \times 1 }$ from the current hidden state $\mathbf { \boldsymbol { s } } _ { i } .$ . Despite their tracking capabilities, equations (2)-(7) form an opaque black box. The dense internal parameter interactions make it difficult to verify how the

LSTM allocates its internal hidden units to adapt to doublyselective channels. This transparency limitation necessitates a diagnostic framework for joint input and internal hidden units filtering without altering the pre-trained model performance, thereby motivating the proposed X-RACE framework.

## III. PROPOSED X-RACE FRAMEWORK

Let U denote the pre-trained LSTM black-box utility model operating as the primary channel estimator, parameterized by S hidden state dimension. To evaluate the time-varying sensitivity of U, the proposed X-RACE framework proceeds according to the following steps, as shown in Figure 1:

1) Joint Noise Mask Generation: First, an interpretability noise LSTM model N is introduced to track the sequential channel inputs and generate a time-varying, subcarrierspecific perturbation mask $z _ { i } ^ { \mathrm { i n } } \in [ 0 , 1 ] ^ { 2 K _ { \mathrm { o n } } \times 1 }$ , and a cell state sensitivity mask $\boldsymbol { z } _ { i } ^ { \mathrm { c e l l } } ~ \in ~ [ 0 , \mathrm { 1 } ] ^ { \boldsymbol { S } \times \mathrm { \bar { 1 } } }$ . In this context, the i-th conventionally estimated channel<sup>1</sup> $\hat { \pmb h } _ { i } ^ { \prime } \in \mathbb { R } ^ { 2 K _ { \mathrm { o n } } \times 1 }$ is fed to the $\mathcal { N }$ model to generate both perturbation masks, such that:

$$
z _ { i } ^ { \mathrm { i n } } , z _ { i } ^ { \mathrm { c e l l } } , s _ { i } ^ { ( \mathcal { N } ) } , c _ { i } ^ { ( \mathcal { N } ) } = \mathcal { N } \left( \hat { h } _ { i } ^ { \prime } , s _ { i - 1 } ^ { ( \mathcal { N } ) } , \mathbf { c } _ { i - 1 } ^ { ( \mathcal { N } ) } ; S \right) .\tag{8}
$$

$\mathbf { \pmb { s } } _ { i } ^ { ( \mathcal { N } ) }$ and $ { \boldsymbol { c } } _ { i } ^ { ( \mathcal { N } ) }$ represent the i-th hidden and cell state vectors of the interpretability LSTM model, respectively. The elements of the generated perturbation masks are bounded within the range [0, 1] via a sigmoid activation.

After that, the generated perturbation masks $z _ { i } ^ { \mathrm { i n } }$ , and $\boldsymbol { z } _ { i } ^ { \mathrm { c e l l } }$ are multiplied by random noise vectors $\epsilon _ { i } ^ { \mathrm { i n } } \sim \mathcal { N } ( 0 , 1 )$ and $\epsilon _ { i } ^ { \mathrm { c e l l } } \sim \mathcal { N } ( 0 , 1 )$ sampled from the standard normal distribution, respectively. The weight noise vectors are then added to the conventional estimated channel and the cell state, such that:

$$
\hat { \pmb { h } } _ { i } ^ { \prime \prime } = \hat { \pmb { h } } _ { i } ^ { \prime } + z _ { i } ^ { \mathrm { i n } } \odot \epsilon _ { i } ^ { \mathrm { i n } } , ~ \pmb { c } _ { i - 1 } ^ { ( \mathcal { U } ) } = \pmb { c } _ { i - 1 } ^ { ( \mathcal { U } ) } + z _ { i } ^ { \mathrm { c e l l } } \odot \pmb { \epsilon } _ { i } ^ { \mathrm { c e l l } } .\tag{9}
$$

2) Utility Model Tracking with Perturbed States: The objective of (8) and (9) is to perturb the input and long-term memory of the pre-trained U model, simultaneously. The frozen utility model U processes the corrupted input sequence using the corrupted cell state memory, such that:

$$
\begin{array} { r } { \hat { \pmb { h } } _ { i } ^ { ( \mathcal { U } ) } , \pmb { s } _ { i } ^ { ( \mathcal { U } ) } , \pmb { c } _ { i } ^ { ( \mathcal { U } ) } = U \left( \hat { \pmb { h } } _ { i } ^ { \prime \prime } , \pmb { s } _ { i - 1 } ^ { ( \mathcal { U } ) } , \pmb { c } _ { i - 1 } ^ { ( \mathcal { U } ) } ; S \right) . } \end{array}\tag{10}
$$

3) Customized Joint Optimization Loss Function: The interpretability model N is trained to quantify the relevance of both input subcarriers and internal hidden units. This is performed by optimizing the learnable $z _ { i } ^ { \mathrm { i n } }$ , and $\boldsymbol { z } _ { i } ^ { \mathrm { c e l l } }$ while minimizing the loss function $\mathcal { L } _ { \mathcal { U } }$ of the pre-trained U model, such that:

$$
\mathcal { L } _ { \mathcal { U } } = \frac { 1 } { N _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t r } } } \left( \tilde { h } _ { i } - \hat { h } _ { i } ^ { ( \mathcal { U } ) } \right) ^ { 2 } .\tag{11}
$$

The input and cell learnable noise masking growth terms $z _ { i } ^ { \mathrm { i n } }$ , and $\boldsymbol { z } _ { i } ^ { \mathrm { c e l l } }$ can be expressed as follows:

<sup>1</sup>The real and imaginary components of the $K _ { \mathrm { o n } }$ active subcarriers are stacked vertically.

![](images/2eb9367150c5680c1464b8e142017d8d8c9f8bc61cb1c8d4cf98ca6e43bdb6aa.jpg)  
Figure 1. Block Diagram of the proposed X-RACE framework.

$$
\mathcal { L } _ { \mathrm { i n p u t } } = \frac { 1 } { 2 K _ { \mathrm { o n } } } \sum _ { j = 1 } ^ { 2 K _ { \mathrm { o n } } } \log \left( z _ { i } ^ { \mathrm { i n } } [ j ] \right) , ~ \mathcal { L } _ { \mathrm { c e l l } } = \frac { 1 } { S } \sum _ { m = 1 } ^ { S } \log \left( z _ { i } ^ { \mathrm { c e l l } } [ m ] \right) .\tag{12}
$$

The training of the $\mathcal { N }$ model aims to minimize a customized joint objective function $\mathcal { L } _ { \mathcal { N } }$ that balances tracking accuracy of $\mathcal { L } _ { \mathcal { U } }$ while maximizing $z _ { i } ^ { \mathrm { i n } }$ and $z _ { i } ^ { \mathrm { c e l l } }$ , weighted by hyperparameters $\lambda _ { 1 }$ and $\lambda _ { 2 } ,$ respectively. Therefore, the objective is to push $z _ { i } ^ { \mathrm { i n } }$ and $z _ { i } ^ { \mathrm { c e l l } }$ toward unity for irrelevant inputs and internal hidden units, such that:

$$
\mathcal { L } _ { \mathcal { N } } = \mathcal { L } _ { \mathcal { U } } - \lambda _ { 1 } \mathcal { L } _ { \mathrm { i n p u t } } - \lambda _ { 2 } \mathcal { L } _ { \mathrm { c e l l } } .\tag{13}
$$

Building upon the learnable $z _ { i } ^ { \mathrm { i n } }$ and $z _ { i } ^ { \mathrm { c e l l } }$ , the X-RACE double optimization problem aims to minimize the Mean Squared Error (MSE) between the true channel, $\tilde { h } _ { i }$ , and the optimized LSTM-based channel estimate, such that:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \tau , P } } & { \displaystyle \mathcal { L } _ { \mathcal { U } ^ { * } } = \frac { 1 } { N _ { \mathrm { t r } } } . \sum _ { i = 1 } ^ { N _ { t r } } \left( \tilde { h } _ { i } - \mathcal { U } ^ { * } \left( \hat { h } _ { i } \odot m _ { \mathrm { i n } } ^ { * } ( \tau ) ; \boldsymbol { c } _ { i } \odot m _ { \mathrm { c e l l } } ^ { * } ( P ) \right) \right) ^ { 2 } , } \\ { \mathrm { s . t . } } & { \displaystyle m _ { \mathrm { u n } } ^ { * } = \mathbb { I } \left( \bar { r } _ { \mathrm { i n } } \ge \tau \right) , \quad \tau \in \mathcal { T } , } \\ & { \displaystyle m _ { \mathrm { c e l l } } ^ { * } = \mathbb { I } \left( \bar { r } _ { \mathrm { c e l l } } \ge \mathrm { P e r c e n t i l e } ( \bar { r } _ { \mathrm { c e l l } } , P ) \right) , P \in \mathcal { P } , } \\ & { \displaystyle \mathrm { B E R } ( \mathcal { U } ^ { * } ) \le \mathrm { B E R } ( \mathcal { U } ) . } \end{array}\tag{14}
$$

$N _ { \mathrm { t r } }$ denotes the number of training samples, and I(·) is the indicator function. The terms $\bar { r } _ { \mathrm { i n } } ~ = ~ 1 - \mathbb { E } [ z ^ { \mathrm { i n } } ]$ and $\bar { r } _ { \mathrm { c e l l } } = 1 - \mathbb { E } [ z ^ { \mathrm { c e l l } } ]$ represent the relevance scores assigned to the U model inputs and internal hidden units, respectively. The relevance is the complement of the induced noise such that higher values reflect greater importance. Furthermore, τ and $P$ denote the optimal input relevance threshold and architectural pruning percentile, defined such that $\mathcal { T } = \{ \tau \in \mathbb { R } \mid \tau _ { \operatorname* { m i n } } \leq$ $\tau \leq \tau _ { \mathrm { m a x } } , \tau = \tau _ { \mathrm { m i n } } + n _ { \tau } \Delta \tau \}$ and $\mathcal { P } = \{ P \in \mathbb { Z } \mid 0 \leq P \leq$ $P _ { \mathrm { m a x } } , P ~ = ~ n _ { P } \Delta P \}$ , with step sizes $\Delta \tau$ and $\Delta P .$ Finally, the constraint ensures that the average BER using the pruned LSTM model $\ b { \mathcal { U } } ^ { * }$ does not exceed that of the full baseline U.

4) Proposed XAI-Assisted Temporal Metrics: To characterize the temporal dynamics of the LSTM memory adaptation, we propose three primary XAI-assisted metrics:

• Saturation Time $( T _ { \mathrm { s a t } } ) \colon$ The earliest temporal index where the LSTM achieves a stable noise update over a window W over I symbols per frame. To mitigate fluctuations, a moving average filter of length $L$ is first applied to smooth the averaged $z _ { i } ^ { \mathrm { i n } }$ , such as:

$$
\tilde { z } _ { i } [ k ] = \frac { 1 } { L } \sum _ { m = - \lfloor L / 2 \rfloor } ^ { \lfloor L / 2 \rfloor } \mathbb { E } \big [ z _ { i + m } ^ { \mathrm { i n } } [ k ] \big ] .\tag{15}
$$

The saturation time $T _ { \mathrm { s a t } }$ is then defined as:

$$
T _ { \mathrm { s a t } } [ k ] = \operatorname* { m i n } _ { i \in [ 1 , I - W ] } : \operatorname* { m a x } _ { \tau \in [ i , i + W - 1 ] } \left| \tilde { z } _ { \tau + 1 } [ k ] - \tilde { z } _ { \tau } [ k ] \right| < \epsilon _ { \mathrm { l o c a l } } [ k ] ,\tag{16}
$$

where $\epsilon _ { \mathrm { l o c a l } }$ sensitivity is dynamically scaled by the subcarrier’s temporal dynamic range to ensure identification robustness:

$$
\epsilon _ { \mathrm { l o c a l } } [ k ] = \epsilon _ { \mathrm { b a s e } } \cdot \left( \operatorname* { m a x } _ { i \in [ 1 , I ] } \tilde { z } _ { i } [ k ] - \operatorname* { m i n } _ { i \in [ 1 , I ] } \tilde { z } _ { i } [ k ] + 1 \right) .\tag{17}
$$

• Importance Drift $( { \bar { \mathcal { D } } } _ { \Phi } ) { \ : \ }$ Measures the average magnitude of the adaptation effort for data and pilot subcarriers, defined as the mean absolute displacement from the initial induced noise at $i = 1$ to the noise at $T _ { \mathrm { s a t } }$ , such that:

$$
\bar { \mathcal { D } } _ { \Phi } = \frac { 1 } { \vert \Phi \vert } \sum _ { k = 1 } ^ { \vert \Phi \vert } \left. \tilde { z } _ { T _ { \mathrm { s a t } } } [ k ] - \tilde { z } _ { 1 } [ k ] \right. , ~ \Phi \in \{ \mathcal { K } _ { \mathrm { d } } , ~ \mathcal { K } _ { \mathrm { p } } \} .\tag{18}
$$

• Relevance Contrast $( \Delta \bar { \mathcal { R } } ) !$ : Quantifies the steady-state discriminative trust gap between data and pilot subcarriers expressed as follows:

$$
\Delta \bar { \mathcal { R } } = \frac { 1 } { | \mathcal { K } _ { \mathrm { d } } | } \sum _ { k = 1 } ^ { K _ { d } } \tilde { z } _ { T _ { \mathrm { s a t } } } [ k ] - \frac { 1 } { | \mathcal { K } _ { \mathrm { p } } | } \sum _ { k = 1 } ^ { K _ { p } } \tilde { z } _ { T _ { \mathrm { s a t } } } [ k ] .\tag{19}
$$

![](images/d9b986618060c5e5a2483a7c00508ac31191fefc7ccd6fac44ff76bafd6dfa3d.jpg)  
Figure 2. Temporal evolution of pilot and data averaged noise weights under varying $F _ { d }$ and modulations, annotated with the proposed XAI metrics.

## IV. SIMULATION RESULTS

The X-RACE framework<sup>2</sup> is evaluated using the DPA-LSTM-NN estimator [7]. Uniform input downsampling [7], SHAP [11], and LIME [10] are employed as benchmark XAI schemes. The IEEE 802.11p standard is used with $K _ { p } = 4$ $K _ { d } = 4 8 , K _ { n } = 1 2$ , and I = 50 OFDM symbols per frame, with a total symbol duration of $T _ { \mathrm { O F D M } } = 8 ~ \mu \mathrm { s }$ . The employed channel models are the high-mobility $( F _ { d } = 1 0 0 0$ Hz) VTV-EX low-frequency selective (LF) and VTV-SDWW highfrequency selective (HF) scenarios [12]. The model is trained on $1 0 ^ { 5 }$ frames using an 80%/20% train-test split. Optimization is performed using the Adam optimizer for 500 epochs with a batch size of 128. For the proposed XAI metrics, we set $L = 3$ and $W = 5$ symbols (10% of the frame) to prevent premature convergence on local fluctuations. We note that $T _ { \mathrm { s a t } }$ is inversely proportional to the $\epsilon _ { \mathrm { b a s e } }$ sensitivity, as tighter tolerances naturally delay convergence. Because the overall conclusions remain consistent regardless of the specific value of $\epsilon _ { \mathrm { b a s e } } .$ , we set $\epsilon _ { \mathrm { b a s e } } = 1 0 ^ { - 3 }$ to balance early convergence and long-term saturation. Finally, the performance analysis is structured across four criteria: (i) LSTM learning dynamics over time selectivity, (ii) the impact of frequency selectivity, (iii) the impact of modulation order, and (iv) a comprehensive XAI and inference computational complexity analysis.

## A. LSTM Learning Dynamics: Time Selectivity Analysis

Figure 2 evaluates $T _ { \mathrm { s a t } } , \ \bar { D } _ { \Phi }$ , and $\Delta \bar { \mathcal { R } }$ across different Doppler frequencies $( F _ { d } )$ and modulations in the LF scenario. Under low Doppler and QPSK modulation, data and pilot subcarriers show similar convergence $( T _ { \mathrm { s a t } } ~ = ~ 5 T$ <sub>OFDM</sub>, $\tilde { z } _ { i } \approx 0 . 5 )$ due to high temporal correlation. Shifting to 64QAM widens ∆R<sup>¯</sup>, as the LSTM prioritizes pilots $( T _ { \mathrm { s a t } } = 6 T _ { \mathrm { O F D M } }$ $\tilde { z } _ { i } ~ \approx ~ 0 . 1 9 )$ while increasingly masking data subcarriers.

At a high Doppler, channel decorrelation forces the LSTM to heavily rely on pilots, as evidenced by reduced $\tilde { z } _ { i }$ and aggressive data subcarrier filtering, with an increased $T _ { \mathrm { s a t } }$ due to relevance uncertainty. Furthermore, higher $F _ { d }$ significantly increases $\bar { \mathcal { D } } _ { \Phi }$ and $\Delta \bar { \mathcal { R } }$ . Thus, the proposed metrics illustrate that the LSTM dynamically adapts its behavior based on its context awareness of the employed scenario.

## B. Impact of Frequency Selectivity

The impact of frequency selectivity is studied under QPSK and $F _ { d } = 1 0 0 0 \ \mathrm { H z } .$ As shown in Figures 3(a)-(b), X-RACE consistently assigns pilots the highest relevance score of 0.9, in both LF and HF scenarios. This proves that pilot relevance is independent of frequency selectivity. However, in the HF scenario, X-RACE assigns fewer data subcarriers a neutral zero score, indicating that increased frequency selectivity demands more informative data subcarriers. Regarding BER performance, X-RACE’s double optimization provides BER performance superiority in the LF scenario using only 5 inputs (threshold 0.3) and 37 internal hidden units, as shown in Figure 3(e). This outperforms SHAP, LIME, and uniform downsampling, which rely on larger relevant subsets without yielding BER performance gains. Conversely, in the HF scenario, X-RACE dynamically scales to 33 relevant inputs (relevance threshold 0.1) and 42 internal hidden units, confirming that severe frequency selectivity necessitates increased relevant subcarriers and model capacity.

## C. Impact of Modulation Order

Figures 3(c)-(d) show that under 64QAM, all XAI methods select pilots as the only relevant inputs. However, X-RACE uniquely maintains uniform, stable relevance scores, whereas LIME and SHAP continue to exhibit score variance. BER performance shows that employing only pilot inputs while retaining the full model architecture successfully matches the unpruned baseline. Conversely, applying architecture or double optimization under 64QAM degrades the BER performance, mainly in the HF scenario as shown in Figure 3(h). Consequently, higher modulation orders allow for aggressive input pruning, but strictly require preserving the original LSTM capacity to preserve the BER performance.

## D. Computational Complexity Analysis

X-RACE delivers a dual computational advantage by minimizing relevance score computation overhead and significantly reducing optimized model complexity. Unlike LIME and SHAP which require expensive iterative perturbations with per-symbol complexities of $\mathcal { O } ( D _ { \mathrm { L I M E } } K _ { \mathrm { o n } } ^ { 2 } )$ and $\mathcal { O } ( D _ { \mathrm { S H A P } } K _ { \mathrm { o n } } )$ respectively, X-RACE derives input and memory relevance masks in a single pass with $\mathcal { O } ( K _ { \mathrm { o n } } + S )$ complexity. Furthermore, as summarized in Table I, X-RACE achieves the best performance-complexity-interpretability trade-off. Compared to the full baseline, it reduces the FLOPs by 78.5% (LF-QPSK) while improving BER, and achieves a 61.5% reduction for 64QAM while preserving BER. Unlike alternative schemes in which complexity reduction degrades BER performance, X-RACE demonstrates a superior interpretability resolution.

![](images/db829f573e71b05c27c4d87fffde96912ec19a4d281e6e2990b9cc39e0bd7bee.jpg)  
(a) LF - QPSK.

![](images/8c86813333b7da7fee15a0dbf16a23e15b06b5418b7d043b62eeaf4e97a486cf.jpg)  
(b) HF - QPSK.

![](images/701dfd963c3e8085b77424987ff00f1d29166ae098e0d0e97ad0c9453030c4d3.jpg)  
(c) LF - 64QAM.

![](images/65913f46c201d0dd59e6018255b2606f8494d0f3fedf491272f0bdca938c4ebb.jpg)  
(d) HF - 64QAM.

![](images/6978ea0166e937c12d45d4d821a3316132a1aa8e5fd1242c5eebf82106d618a4.jpg)  
(e) LF - QPSK.

![](images/1626884f0a8bcb721469d84305c4ccb57bad48da6bb3849cf01dfe3fe9a20015.jpg)  
(f) HF - QPSK.

![](images/fb928f175a29d06b39bfa1c575148bd5cbd79d5f953895643ffe802b87f60568.jpg)  
(g) LF - 64QAM.

![](images/0b94d6b3c2cb58ce1660aede069c97f5c7b3e8e4a16e9c433ab160b0dcffbb36.jpg)  
(h) HF - 64QAM.  
Figure 3. Relevance score distributions (top) and BER performance (bottom) of the proposed X-RACE framework and benchmarked XAI schemes. For all evaluated schemes, the BER curves correspond to the minimum achieved BER with respect to the best relevance threshold.

Table I  
XAI-ASSISTED ARCHITECTURAL COMPLEXITY REDUCTION IN TERMS OFFLOATING-POINT OPERATIONS (FLOPS).
<table><tr><td rowspan=1 colspan=1>XAIOptimization</td><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Architecture $( K _ { 0 \mathrm { n } } \cdot \mathrm { S } )$ </td><td rowspan=1 colspan=1>FLOPs</td><td rowspan=1 colspan=1>Reduction</td></tr><tr><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>(52-52)</td><td rowspan=1 colspan=1>64,896</td><td rowspan=1 colspan=1>*</td></tr><tr><td rowspan=1 colspan=1>UniformDownsampling</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>(28-28)</td><td rowspan=1 colspan=1>18,816</td><td rowspan=1 colspan=1>71%</td></tr><tr><td rowspan=2 colspan=1>X-RACE</td><td rowspan=1 colspan=1>LF-QPSK</td><td rowspan=1 colspan=1>(5-37)</td><td rowspan=1 colspan=1>13,912</td><td rowspan=1 colspan=1>78.5%</td></tr><tr><td rowspan=1 colspan=1>HF-QPSK</td><td rowspan=1 colspan=1>(33-42)</td><td rowspan=1 colspan=1>36,288</td><td rowspan=1 colspan=1>44.1%</td></tr><tr><td rowspan=2 colspan=1>SHAP</td><td rowspan=1 colspan=1>LF-QPSK</td><td rowspan=1 colspan=1>(14-52)</td><td rowspan=1 colspan=1>33,280</td><td rowspan=1 colspan=1>48.7%</td></tr><tr><td rowspan=1 colspan=1>HF-QPSK</td><td rowspan=1 colspan=1>(11-52)</td><td rowspan=1 colspan=1>30,784</td><td rowspan=1 colspan=1>52.6%</td></tr><tr><td rowspan=2 colspan=1>LIME</td><td rowspan=1 colspan=1>LF-QPSK</td><td rowspan=1 colspan=1>(5-52)</td><td rowspan=1 colspan=1>25,792</td><td rowspan=1 colspan=1>60.3%</td></tr><tr><td rowspan=1 colspan=1>HF-QPSK</td><td rowspan=1 colspan=1>(8-52)</td><td rowspan=1 colspan=1>28,288</td><td rowspan=1 colspan=1>56.4%</td></tr></table>

## V. CONCLUSION

This paper introduced X-RACE, a dual-optimization attribution framework that resolves LSTM opacity and overhead in doubly-selective channels by simultaneously pruning input subcarriers and internal hidden units. Furthermore, novel temporal XAI metrics uncovered the LSTM’s learning dynamics. Simulations confirm that X-RACE reduces inference complexity by at least 44.1% while improving BER performance, achieving a superior performance-complexity-interpretability trade-off over classical XAI. Future work will extend the framework to incorporate gradient-based XAI and optimize other RNN-based channel estimators.

## REFERENCES

[1] S. Majumdar, Q. Wei, S. Schwarzmann, R. Trivisonno, and G. Carle, “Toward AI-Native 6G Systems: Standards Enablers for 6G Network Automation,” IEEE Communications Standards Magazine, vol. 10, no. 1, pp. 145–153, 2026.

[2] M. A. Ferrag, A. Lakas, and M. Debbah, “6G Needs Agents: Toward Agentic AI-Native Networks for Autonomous Intelligence,” IEEE Open Journal of the Communications Society, vol. 7, pp. 7254–7282, 2026.

[3] W. Saad, O. Hashash, C. K. Thomas, C. Chaccour, M. Debbah, N. Mandayam, and Z. Han, “Artificial General Intelligence (AGI)-Native Wireless Systems: A Journey Beyond 6G,” Proceedings of the IEEE, vol. 113, no. 9, pp. 849–887, 2025.

[4] K. Kandali and S. Nouh, “AI-Native V2X and Internet of Vehicles Systems: Architectures, Learning Paradigms, and Open Challenges,” IEEE Open Journal of Intelligent Transportation Systems, vol. 7, pp. 1775–1788, 2026.

[5] A. F. d. Reis, B. S. Chang, Y. Medjahdi, G. Brante, and F. Bader, “LSTM-Based Time-Frequency Domain Channel Estimation for OTFS Modulation,” IEEE Transactions on Vehicular Technology, vol. 73, no. 10, pp. 15 049–15 060, 2024.

[6] A. K. Gizzini, Y. Medjahdi, A. J. Ghandour, and L. Clavier, “Explainable AI for Enhancing Efficiency of DL-based Channel Estimation,” IEEE Transactions on Machine Learning in Communications and Networking, 2025.

[7] A. F. Dos Reis, Y. Medjahdi, B. S. Chang, J. Sublime, G. Brante, and C. F. Bader, “Low Complexity LSTM-NN-Based Receiver for Vehicular Communications in the Presence of High-Power Amplifier Distortions,” IEEE Access, vol. 10, pp. 121 985–122 000, 2022.

[8] H. Sun, Y. Liu, A. Al-Tahmeesschi, A. Nag, M. Soleimanpour, B. Canberk, H. Arslan, and H. Ahmadi, “Advancing 6G: Survey for Explainable AI on Communications and Network Slicing,” IEEE Open Journal of the Communications Society, vol. 6, pp. 1372–1412, 2025.

[9] T. Senevirathna, V. H. La, S. Marcha, B. Siniarski, M. Liyanage, and S. Wang, “A Survey on XAI for 5G and Beyond Security: Technical Aspects, Challenges and Research Directions,” IEEE Communications Surveys & Tutorials, vol. 27, no. 2, pp. 941–973, 2025.

[10] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why Should I Trust You? Explaining the Predictions of Any Classifier,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1135–1144.

[11] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 4768–4777.

[12] I. Sen and D. W. Matolak, “Vehicle–Vehicle Channel Models for the 5- GHz Band,” IEEE Transactions on Intelligent Transportation Systems, vol. 9, no. 2, pp. 235–245, 2008.