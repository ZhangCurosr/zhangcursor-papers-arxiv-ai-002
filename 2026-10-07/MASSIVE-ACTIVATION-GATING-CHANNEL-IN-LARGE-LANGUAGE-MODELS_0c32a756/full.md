# MASSIVE ACTIVATION GATING CHANNEL IN LARGE LANGUAGE MODELS

Minjia Mao<sup>1</sup>, Shi Chen<sup>1</sup>, Bowen Yin<sup>2</sup>, Xiao Fang<sup>1</sup>

<sup>1</sup> University of Delaware

<sup>2</sup> Peking University

{mjmao,shichen,xfang}@udel.edu bowenyin@stu.pku.edu.cn

## ABSTRACT

Massive activations, a phenomenon in which a small number of hidden channels exhibit exceptionally large magnitudes, are pervasive in large language models (LLMs). However, the mechanism by which a token develops massive activations as it propagates through a pretrained LLM remains poorly understood. In this paper, we find that the emergence of massive activations is controlled by a single channel in the input embedding to a spike feed-forward network (FFN). The position of this channel is fixed for a particular LLM. We name this channel the massive activation gating channel (MAGC). When the value of the MAGC is sufficiently large (or small, depending on the LLM), the output of the spike FFN exhibits massive activations. Examining six LLMs across four model families and different model sizes, we verify the existence and effect of MAGC. We further provide a theoretical explanation of the mechanism by which MAGC induces massive activations. When the value of MAGC is sufficiently large (or small), the output of a spike FFN asymptotically reduces to a quadratic form that mixes a few columns of the down-projection matrix of the FFN. Since these columns exhibit the shape of massive activations, the output therefore exhibits massive activations.

## 1 INTRODUCTION

Large Language Models (LLMs) have demonstrated strong capabilities on various tasks (Bubeck et al., 2023). While the majority of existing work focuses on external behaviors of LLMs, it is also important to understand the internal mechanisms of LLMs. Among them, attention sink (Xiao et al., 2023) and massive activations (Sun et al., 2024) have been found in modern LLMs across different model families. Specifically, attention sink refers to the phenomenon in which the first token (or other tokens with little semantic importance) in a sequence receives a disproportionately large amount of attention from subsequent tokens (Yu et al., 2024; Barbero et al., 2025), typically in the middle transformer blocks. The cause of attention sink is usually attributed to massive activations (Sun et al., 2024; Yan et al., 2024), where a few channels (dimensions) in the hidden representation of the first token (or other tokens with little semantic importance) exhibit large activation magnitudes before attention computation. It is important to unveil the mechanism underlying massive activations and attention sink, as these phenomena have been widely employed in various critical tasks, including KV-cache optimization (Ge et al., 2023), inference (Xiao et al., 2023), quantization (Son et al., 2024), and hallucination mitigation (Liu et al., 2026).

In this study, we focus on massive activations. Although existing research has extensively demonstrated the phenomenon of massive activations, research on the mechanism underlying massive activations remains limited. We address this problem by asking: how does a token develop massive activations as it propagates a pretrained transformer? We refer to tokens that exhibit massive activations as spike tokens (Sun et al., 2026).

As LLMs are powered by billions of parameters, our analysis draws attention to the computation of each sub-module (attention and feed-forward networks) in transformers. Existing work (Zhang et al., 2026; Sun et al., 2026) has demonstrated that massive activations emerge after certain feed-forward networks (FFNs) and are maintained in the middle transformer blocks through the residual layer (see Section 2 for more details). For example, in LLaMA-2-7B, massive activations of a spike token emerge after the FFN in the second transformer block (Sun et al., 2024; 2026) and are maintained until reaching the last FFN. We refer to the FFN that produces massive activations as the spike FFN. The magnitudes of massive activations remain similar in the middle transformer blocks, indicating that only the residual layer maintains massive activations and that no further massive activations are triggered during the middle computation.

Sun et al. (2026) assume that the SiLU activation function in the spike FFN behaves as a near-identity function for spike tokens. Under this assumption, each output channel k of the spike FFN can be expressed as a quadratic form characterized by a channel-specific coefficient matrix $S _ { k }$ . The magnitude of output channel k is determined by the dominant eigenvalue of $S _ { k }$ . Since the dominant eigenvalues associated with a few channels are substantially larger than those of other channels, massive activation emerges.

However, our empirical results show that the near-identity assumption on which Sun et al. (2026) rely to explain massive activations is only valid for a small number of models. Among the six models examined in this study, only one model approximately satisfies the assumption (see Appendix A for more details). Moreover, their explanation overlooks the properties of input embeddings to the spike FFN. Instead, we find that the emergence of massive activations is controlled by a single channel in the input embeddings of a spike token to the spike FFN: massive activations appear once the value of this channel is sufficiently large (or small, depending on the LLM). For a given LLM, the position of this channel is fixed. We refer to this channel as the massive activation gating channel (MAGC). Unlike massive activations, whose magnitudes can reach approximately 400 in LLaMA-3.2-1B, the MAGC in the embedding of a spike token has an average value of only around 8.8. Although this value is substantially larger than those of the other channels in the embedding, its magnitude remains moderate because the embedding is layer-normalized before being fed into the FFN.

We verify this finding across multiple decoder-only LLM families, including LLaMA, Qwen, DeepSeek, and Mistral, with model sizes ranging from 1B to 32B parameters. Moreover, our theoretical analysis shows that massive activations can be attributed to the alignment between MAGC and certain rows and columns of the spike FFN weight matrices. Similarly, we empirically demonstrate that this alignment occurs in different LLMs. We summarize our contributions as follows.

1. The emergence of massive activations is controlled by a single channel in the input embedding to a spike FFN, and the position of this channel is fixed for a particular LLM. We name this channel the massive activation gating channel (MAGC). When the value of the MAGC is sufficiently large (or small, depending on the LLM), the output of the spike FFN exhibits massive activations. Examining six LLMs across four model families and different model sizes, we verify the existence and effect of MAGC.

2. We provide a theoretical explanation of the mechanism by which MAGC induces massive activations. When the value of MAGC is sufficiently large (or small), the output of a spike FFN asymptotically reduces to a quadratic form that mixes a few columns of the down-projection matrix of the FFN. Since these columns exhibit the shape of massive activations, the output therefore exhibits massive activations.

## 2 TRANSFORMER ARCHITECTURE

A dense transformer model consists of L transformer blocks, each of which can be further decomposed into modules. Taking LLaMA as an example (Touvron et al., 2023), denote the input to the l-th transformer block as $\breve { X } ^ { l } = \langle { \pmb x } _ { 0 } ^ { l } , { \pmb x } _ { 1 } ^ { l } , \cdot \cdot \cdot , { \pmb x } _ { T } ^ { l } \rangle$ . Without further specification, $\cdot _ { t } ^ { l }$ denotes an embedding of the t-th token in the l-th transformer block. Since our following analysis focuses on the single spike FFN, we omit the superscript l for notational simplicity. Each block of LLaMA consists of an attention sub-module and an FFN sub-module, each of which uses pre-normalization (RMSNorm (Zhang & Sennrich, 2019)) and includes a residual layer after its computation.

The input embedding is first normalized by an RMSNorm, referred to as input layer norm. Formally, if $\pmb { x } _ { t } = \langle x _ { t } ( 0 ) , x _ { t } ( 1 ) , \cdot \cdot \cdot , x _ { t } ( d - 1 ) \rangle$ ⟩, where d is the dimension of the embedding, then

$$
n _ { t } = { \mathrm { R M S N o r m } } ( \boldsymbol { x } _ { t } ) : n _ { t } ( j ) = \frac { \boldsymbol { x } _ { t } ( j ) \cdot \boldsymbol { w } _ { j } } { \mathrm { R M S } ( \boldsymbol { x } _ { t } ) } , \mathrm { R M S } ( \boldsymbol { x } _ { t } ) = \sqrt { \frac { \sum _ { j } x _ { t } ( j ) ^ { 2 } } { d } + \epsilon } ,\tag{1}
$$

where $w _ { j }$ are trainable parameters and ϵ is a predefined small number. We denote $\cdot ( j )$ as the j-th channel of the embedding. Next, the normalized embedding $\mathbf { \nabla } _ { \mathbf { \eta } _ { n _ { t } } }$ is transformed by multi-head attention (MHA, or more specifically, grouped-query attention (Ainslie et al., 2023)):

$$
( \pmb { u } _ { 0 } , \pmb { u } _ { 1 } , \cdots , \pmb { u } _ { T } ) = \mathrm { M H A } ( \pmb { n } _ { 0 } , \pmb { n } _ { 1 } , \cdots , \pmb { n } _ { T } ) ,
$$

where $\mathbf { \Delta } \mathbf { u } _ { t }$ is the embedding output of MHA. Usually, a transformer includes a residual layer after the multi-head attention (Vaswani et al., 2017). Next, the FFN sub-module follows a similar configuration with another RMSNorm of different parameters.

$$
\begin{array} { r } { \pmb { h } _ { t } = \mathbf { R M S N o r m } ( \pmb { x } _ { t } + \pmb { u } _ { t } ) . } \end{array}
$$

A feed-forward network (FFN) is then applied to $\boldsymbol { h } _ { t } .$ . In LLaMA, the FFN consists of a SwiGLU gated activation function (Shazeer, 2020) with three weight matrices:

$$
y _ { t } = \mathrm { F F N } ( h _ { t } ) , y _ { t } = W _ { \mathrm { d o w n } } \cdot \left( \mathrm { S i L U } \left( W _ { \mathrm { g a t e } } h _ { t } \right) \odot \left( W _ { \mathrm { u p } } h _ { t } \right) \right) ,\tag{2}
$$

where ⊙ is the element-wise product and y is the output of FFN. $W _ { \mathrm { g a t e } } , W _ { \mathsf { u p } } \in \mathbb { R } ^ { m \times d }$ and $W _ { \mathrm { d o w n } } \in$ $\mathbb { R } ^ { d \times m }$ are pretrained weight matrices, where $m > d$ is the FFN intermediate dimension. Finally, another residual layer follows the FFN. That is, the output of a transformer block is ${ \pmb x } _ { t } + { \pmb u } _ { t } + { \pmb y } _ { t }$

![](images/3c07dbc352e11628bf059e296f1facb79ef44b1669ce55c1313e4b4877c32a60.jpg)  
(a) Value of input h to the spike FFN. We highlight the massive activation gating channel (894) in red.

![](images/a2e5ccc03a1e050793d5f801518e720119b593a42714fe97fe6e63842e8b2b1a.jpg)  
(b) Magnitude of output y of the spike FFN. Token 0 exhibits massive activations. We depict magnitudes instead of actual values for better visualization.  
Figure 1: Inputs and Outputs of the Spike FFN.

## 3 MASSIVE ACTIVATION GATING CHANNEL

Experimental Setup. In this section, we utilize a small but advanced LLM, LLaMA-3.2-1B (Dubey et al., 2024). We verify the findings discovered in this section on other model families with larger model sizes in Section 5.1. In LLaMA-3.2-1B, d = 2, 048 and $m = 8 , 1 9 2$ . We generate token sequences (without the BOS token) completely randomly to avoid the influence of semantics. Specifically, we randomly sample five tokens in the vocabulary set V and construct an input token sequence to the LLM. Note that the input tokens can be multilingual as well as completely meaningless. We conduct this sampling 5 times.

## 3.1 EXISTENCE OF MAGC IN THE INPUT TO SPIKE FFN

Following Sun et al. (2026), we demonstrate the top-3 channel magnitudes of the transformer block outputs ${ \pmb x } + { \pmb u } + { \pmb y }$ in Figure A1 in Appendix B. Similarly, we show that massive activations are first triggered by the FFN in an early layer, and are then preserved and propagated along the residual stream across layers. As depicted, we identify that in LLaMA-3.2-1B, the FFN in the second transformer block (i.e., block 1) produces massive activations. We name this FFN the spike FFN and focus our analysis on it.

A straightforward method for analyzing the causes of massive activations is to trace their inputs. To this end, we directly visualize the embeddings of the inputs and outputs of the spike FFN for different tokens in Figure 1, using a token sequence example. We demonstrate other examples in Figure A2 in Appendix B.2. As shown in Figure 1b, the first token (“stejne”) exhibits massive activations, whereasˇ other tokens do not. Comparing their inputs to the FFN, we find that the input h for the first token has a single outlier value, as shown in Figure 1a. Specifically, the 894th channel of the first token has a value of 8.807, while the absolute values of the other channels are all less than 2. The 894th channel of the other tokens has values of 0.814, -0.763, 0.335, and 1.029, respectively. In the next subsection, we will show that this single channel controls the emergence of massive activations, and we name this channel the massive activation gating channel (MAGC).

Block 1 FFN output (peak@ch 0, a=1)  
Block 1 FFN output (peak@ch 1742, a=1)  
![](images/21448555bc7ccc7468b72272933ef06a4149ad2ea0205f3f87148a66252bc021.jpg)

![](images/a576f2f48196ccaeeda45c4ebe7345f26a903f2de9df51324090978fd3374825.jpg)

![](images/830ecf4f41912398ef947746c0f5b1d9e0ac26790da2d45d7b86d3c24b0170e3.jpg)

![](images/46556ad61a374f77b563f6cb80080b2275b9f9e5cfbeadf174da4497ea9f9b24.jpg)

![](images/ce3f16d3601cac0b562cabee50e05eed05bd2e69c4a8d308de0b961a52c1c16a.jpg)

![](images/e92e38c36803424dde430fae98b6524625364a47654479c83d32fb81c9c3949b.jpg)  
Figure 2: From top to bottom, the figure shows the output of the spike FFN when the input vector values are randomly sampled from a Pareto distribution (a = 1), with the input value of the 894th (MAGC), 0th, or 1742nd channel swapped with the maximum input value, respectively.

## 3.2 MAGC TRIGGERS MASSIVE ACTIVATIONS

Next, we show that this single-channel spike triggers massive activations through the computation of the spike FFN. To empirically validate this claim, we conduct the following experiment. Specifically, we randomly generate the input embedding to the spike FFN from a Pareto distribution with random signs: each channel value of the input embedding is i.i.d. drawn as $X ~ = ~ \mathrm { r a n d o m } ( \pm ) Z$ with $Z \sim \mathrm { P a r e t o } ( a ) , Z \geq 0 .$ , where each element takes +Z and −Z with equal probability 0.5, and a is a fixed shape parameter. Next, we swap the input value of the 894th channel with the maximum input value. We then input this randomly generated embedding into the spike FFN. Meanwhile, we conduct a controlled experiment. Specifically, we used the same randomly sampled input embedding but swapped the input value of a randomly selected channel other than the 894th, such as the 0th or 1742nd, with the maximum value. We report the results for $a = 1$ in this section and present the results for $a = 3$ and $a = 5$ in Appendix B.3.

Figure 2 depicts the randomly generated inputs and their corresponding outputs after the spike FFN. It can be observed that when swapping the value of the 894th channel with the maximum value, the outputs of the spike FFN demonstrate massive activations. Since the randomly generated inputs carry no semantic meaning, this phenomenon mainly indicates that the FFN learns to recognize MAGC. However, swapping the value of any other channel, such the 0th or 1742nd channel, with the maximum value does not lead to the output of massive activations. We therefore conclude that MAGC controls the emergence of massive activations.

Moreover, as shown in Figure A3 and A4 in Appendix B.3, decreasing a increases the maximum value generated by the Pareto distribution. More importantly, assigning a larger maximum value to the 894th channel of the spike FFN input results in a correspondingly larger activation magnitude in its output. Specifically, as a decreases from 5 to 1, the largest output activation magnitude increases from approximately 40 to 400. These results demonstrate that the spike FFN is highly sensitive to the value of the 894th input channel, with a larger value in this channel inducing more “massive” activations.

Key Takeaway 1: There exists a massive activation gating channel (MAGC) in the hidden representation of a spike token, whose position is fixed for a specific LLM. When the hidden representation is used as the input to the spike FFN, the FFN output exhibits massive activations when the value of the MAGC in the hidden representation is sufficiently large (or small, depending on the LLM).

## 4 FROM MAGC TO MASSIVE ACTIVATIONS

In this section, we examine how a large value of MAGC induces massive activations in LLMs. Since a large value of MAGC corresponds to a large magnitude in a certain channel of the input to the FFN, we begin by analyzing the asymptotic property of the FFN with respect to any input channel. Recall the FFN computation in Equation $2 \colon \dot { \boldsymbol { y } } ( \dot { h } ) \stackrel { \sim } { = } W _ { \mathrm { d o w n } } \left( \mathrm { S i L U } ( W _ { \mathrm { g a t e } } \acute { h } ) \odot \bar { W _ { \mathrm { u p } } } \dot { h } \right)$ . We show that, when a fixed channel has a large magnitude (in either direction), the mechanism of the FFN can be simplified as mixing columns of $\bar { W } _ { \mathrm { d o w n } }$ . Note that the mixing mechanism depends on the direction of the channel value $( x \to + \infty \mathrm { o r } x \to - \infty )$ , which explains why MAGC also has a fixed gating direction.

## 4.1 ASYMPTOTIC PROPERTY OF AN FFN W.R.T. AN INPUT CHANNEL

Figure 1a demonstrates the phenomenon that the input to the FFN contains a large magnitude in a certain channel for spike tokens. We begin by analyzing the asymptotic property of any FFN with respect to any input channel. Consider a gated FFN in Equation 2, for input $\bar { h } \in \mathbb { R } ^ { d }$

$$
\mathrm { F F N } ( h ) = y ( h ) = W _ { \mathrm { d o w n } } \left( \mathrm { S i L U } ( W _ { \mathrm { g a t e } } h ) \odot W _ { \mathrm { u p } } h \right) .\tag{3}
$$

where $W _ { \mathrm { u p } } \in \mathbb { R } ^ { m \times d } , W _ { \mathrm { g a t e } } \in \mathbb { R } ^ { m \times d }$ , and $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { d \times m }$ are parameter matrices, $\mathrm { S i L U } ( x ) : =$ $x \cdot \sigma ( x )$ with the logistic sigmoid function $\begin{array} { r } { \sigma ( x ) : = \frac { 1 } { 1 + \exp ( - x ) } } \end{array}$ , and ⊙ is the element-wise product.

We focus on the input h controlled by a fixed single channel i. For any value $x \in \mathbb { R }$ in the fixed channel i, the input can be written as,

$$
h ( x ) = x \cdot e _ { i } + \bar { h } .\tag{4}
$$

Here, $\bar { \boldsymbol { h } } \in \mathbb { R } ^ { d }$ is the representation outside the focused channel i with $\bar { \mathbf { h } } ( i ) = 0$ , and $e _ { i }$ is the unit vector at position i. Next, we analyze the asymptotic property of ${ \pmb y } ( { \pmb h } ( x ) )$ for $x \to + \infty$ without

loss of generality. Let $\omega _ { j } = W _ { \mathrm { d o w n } } ^ { ( : , j ) } \Big / \Big \| W _ { \mathrm { d o w n } } ^ { ( : , j ) } \Big \|$ be the normalized j-th column of $W _ { \mathrm { d o w n } }$ . The 2   
following theorem shows the closed form of $y ( \tilde { h ( x ) } )$ when $x \to + \infty$

Theorem 4.1. Given input $\pmb { h } ( x ) = x \cdot \pmb { e } _ { i } + \pmb { \bar { h } } ,$ , when $x \to + \infty ,$ , the output ofan FFN has a quadratic form ${ \pmb y } ( { \pmb h } ( x ) ) = x ^ { 2 } \cdot { \pmb y } ^ { * } + o ( x ^ { 2 } )$ with constant $\boldsymbol { y } ^ { * } \in \mathbb { R } ^ { d }$ equal to

$$
{ \pmb y } ^ { * } = \sum _ { j = 1 } ^ { m } { \boldsymbol \alpha } _ { j } \cdot { \boldsymbol \omega } _ { j } ,\tag{5}
$$

where $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { m } )$ is the combination ofcoefficients in the weight matrices:

$$
\alpha _ { j } = \left\{ \begin{array} { l l } { W _ { \mathrm { g a t e } } ^ { ( j , i ) } W _ { \mathrm { u p } } ^ { ( j , i ) } \left. W _ { \mathrm { d o w n } } ^ { ( : , j ) } \right. _ { 2 } } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } > 0 } \\ { 0 } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } \le 0 } \end{array} . \right.\tag{6}
$$

Proof. See Appendix C.1.

Specifically, for any input channel i in $h , { \mathrm { S i L U } } ( W _ { \mathrm { g a t e } } h )$ acts as a gating function, choosing column j only when $W _ { \mathrm { g a t e } } ^ { ( j , i ) } x > 0$ , and $\alpha _ { j }$ controls the sensitivity to each $\omega _ { j }$ . From Theorem 4.1, when x is large enough, the output $\pmb { y } ( \pmb { h } ( x ) ) = x ^ { 2 } \cdot \pmb { y } ^ { * } + o ( x ^ { 2 } )$ exhibits a pattern of massive activations only if ${ \pmb y } ^ { * }$ has the same pattern. Similarly, an analogous result holds as $x \to - \infty$

## 4.2 FROM MAGC TO MASSIVE ACTIVATIONS

We introduce several metrics below to quantify the degree of massive activations. Since massive activations describe the phenomenon in which certain activations exhibit large magnitudes, we utilize the energy distribution to characterize their magnitude distribution.

Definition 4.2 (Energy Distribution). Given vector $\pmb { v } = ( v _ { 1 } , \dots , v _ { d } ) \in \mathbb { R } ^ { d }$ , let its energy distribution be $\pmb { p } ( \pmb { v } ) = ( p _ { 1 } ( \pmb { v } ) , \dots , p _ { d } ( \pmb { v } ) )$ ), where

$$
p _ { i } ( \pmb { v } ) = \frac { v _ { i } ^ { 2 } } { \lVert \pmb { v } \rVert _ { 2 } ^ { 2 } } , \quad \forall i = 1 , \ldots , d .\tag{7}
$$

The energy distribution is a probability distribution. Moreover, if y exhibits massive activations, ${ \pmb p } ( \pmb y )$ should be highly concentrated. We define the degree of concentration as the negative entropy of ${ \pmb p } ( \pmb y )$ plus a constant. Formally,

Definition 4.3 (Concentration Degree). Given vector $\pmb { v } \in \mathbb { R } ^ { d }$ , let its concentration degree be

$$
R ( { \pmb v } ) = \log d + \sum _ { i = 1 } ^ { d } p _ { i } ( { \pmb v } ) \log p _ { i } ( { \pmb v } )\tag{8}
$$

where $\pmb { p } ( \pmb { v } ) = ( p _ { 1 } ( \pmb { v } ) , \dots , p _ { d } ( \pmb { v } ) )$ is the energy distribution defined in Definition 4.2.

The concentration degree $R ( { \pmb v } ) \in [ 0 , \log d ]$ is exactly the negative entropy plus a constant log d. A larger $R ( v )$ means v is more concentrated. $R ( \pmb { v } ) = 0$ means v is uniform, while $R ( { \boldsymbol { v } } ) = \log d$ means v has only one non-zero entry.

We now fix MAGC as the focused channel $i ^ { * }$ and analyze the relationship between the concentration degree of the output $\boldsymbol { y } ^ { * }$ and those of $\omega _ { j }$ and $\alpha .$ . We first find that the following assumption is empirically satisfied.

Assumption 4.4. When MAGC is the focused channel, α has a high concentration degree, i.e., $R ( \alpha ) \bar { \geq } \log m - \epsilon$ for a small ϵ. Let $j ^ { * } = \arg \operatorname* { m a x } _ { j } p _ { j } ( \pmb { \alpha } )$ be the dominant column index; then

$$
R ( \pmb { \alpha } ) \geq \log m - \epsilon \Rightarrow p _ { j ^ { * } } ( \pmb { \alpha } ) \geq e ^ { - \epsilon } .\tag{9}
$$

Empirical verification. Rather than assuming this property, we verify it directly by scanning all $d = 2 { , } 0 4 8$ input channels of the spike FFN in LLaMA-3.2-1B: for each channel i we form the coefficient vector α from Theorem 4.1 and measure its concentration $R ( \alpha )$ , dominant energy share $p _ { j ^ { \ast } } ( \alpha )$ , and norm $\| \pmb { \alpha } \| _ { 2 } . \ \mathbf { M A G C } \ ( i ^ { * } = 8 9 4 )$ attains the largest concentration among all channels, with $R ( \pmb { \alpha } ) = 9 . 0 0 9$ against log $m = 9 . 0 1 1 ( { \mathrm { i . e . } } \epsilon \approx 1 . 9 \times 1 0 ^ { - 3 } )$ and $p _ { j ^ { * } } ( \pmb { \alpha } ) = 0 . 9 9 9 8$ , whereas the mean over all channels is only $R ( \pmb { \alpha } ) = 2 . 9 0 8$ . This confirms that MAGC is a significant outlier in α-concentration.

![](images/b8be719c4cb82d7d1f43bb4dab56d6a0e9b1b6b237afe91ad6866f2420607001.jpg)  
(a) α vector when $i ^ { * } = 8 9 4$ for $\mathrm { L L a M A } { - } 3 . 2 { - } 1 \mathrm { B }$

![](images/02fa354f8f5be68d67d11c336c2ecb97baa0a1bd979bc367d52d9381959f8cac.jpg)  
(b) $\omega _ { 1 4 1 7 }$ for LLaMA-3.2-1B.

![](images/008a2474194ceac84509a9a8b62923614e905da6cd09e8ee53134ea52758f291.jpg)  
(c) $\omega _ { \mathrm { 0 } }$ for LLaMA-3.2-1B.

![](images/4b637aef9a0b7d9d9968f9dfbb7902b5a908b1a6fecb357b4079ccfaddc8779a.jpg)  
(d) $\omega _ { 1 }$ for LLaMA-3.2-1B.  
Figure 3: Visualization results for LLaMA-3.2-1B.

A high concentration degree $R ( \alpha )$ indicates that a few $p _ { j } ( \pmb { \alpha } )$ are extremely large. We empirically demonstrate α when $i ^ { * } = 8 9 4$ for $\mathbf { L L a M A - } 3 . 2 \mathbf { - } 1 \mathbf { B }$ in Figure 3a and highlight the largest magnitude in the vector. As shown, α is highly concentrated with $\bar { j } ^ { * } = 1 4 1 7$ . That is, the output y of the FFN is largely controlled by the $j ^ { * }$ -th column of $W _ { \mathrm { d o w n } }$ . Theoretically, based on the high concentration of α, we next quantify the contribution of the remaining columns.

Definition 4.5. Given $j ^ { * } = \arg \operatorname* { m a x } _ { j } p _ { j } ( \pmb { \alpha } )$ , define the residual amplification factor

$$
\kappa = \frac { \left\| \sum _ { j \neq j ^ { * } } \alpha _ { j } \omega _ { j } \right\| _ { 2 } } { \sqrt { \sum _ { j \neq j ^ { * } } \alpha _ { j } ^ { 2 } } } .
$$

The residual amplification factor κ quantifies the contribution of other channels after removing the dominant column $j ^ { * }$ . In the following proposition, κ controls how strongly the non-dominant columns can perturb the energy distribution of the output.

## Proposition 4.6. Let

$$
\tau = \kappa \sqrt { \frac { 1 - p _ { j ^ { * } } ( \alpha ) } { p _ { j ^ { * } } ( \alpha ) } } .\tag{10}
$$

We have

$$
| R ( \pmb { y } ^ { \ast } ) - R ( \pmb { \omega } _ { j ^ { \ast } } ) | \leq \Delta \left( | \sin \angle ( \pmb { y } ^ { \ast } , \pmb { \omega } _ { j ^ { \ast } } ) | \right) \leq \Delta \left( \tau \right) ,\tag{11}
$$

where $f o r t \in \left( 0 , 1 - \textstyle { \frac { 1 } { d } } \right)$ , the function $\Delta$ is defined as $\begin{array} { r } { \Delta ( t ) = - t \log t - ( 1 - t ) \log ( 1 - t ) + t \log ( d - \frac { 1 } { d - 1 } ) + \log t - 1 } \end{array}$ 1).

Proof. See Appendix C.2.

Proposition 4.6 indicates that when $p _ { j ^ { \ast } } ( \alpha )$ is large (i.e., close to 1), we have $\tau \approx 0$ , and there is almost no gap between the concentration degree of $\boldsymbol { y } ^ { * }$ and that of $\omega _ { j ^ { * } }$

In the case of LLaMA-3.2-1B, the output y of the FFN is largely controlled by the $j ^ { * }$ -th (1417th) column of $W _ { \mathrm { d o w n } }$ . We next depict $\omega _ { j }$ ∗ in Figure 3b. Comparing Figure 3b to 1b, we find that these two vectors are highly co-linear, empirically with | sin $\mathcal { L } ( \pmb { y } ^ { * } , \bar { \pmb { \omega } } _ { j ^ { * } } ) \bar { \vert } = 0 . 0 0 5 5$ . For other columns such as columns 0 and 1 in Figures 3c and 3d, the parameters have lower maximum magnitudes and do not contain the pattern of massive activations. Note that both $\boldsymbol { y } ^ { * }$ and $\omega _ { j ^ { \ast } }$ ∗ are computed only from the pretrained weight matrices. We show that when the value of channel $i ^ { * }$ is large enough, the actual output y is dominated by $\omega _ { j ^ { \ast } }$ , which leads to $\pmb { y } \approx x ^ { 2 } \pmb { y } ^ { * } \approx x ^ { 2 } \alpha _ { j ^ { * } } \omega _ { j ^ { * } }$ . If $\alpha _ { j }$ ∗ has a large magnitude and $\omega _ { j ^ { \ast } }$ ∗ is highly concentrated (i.e., contains the pattern of massive activations), the output y should exhibit the same pattern. Below, we summarize the mechanism by which the MAGC in the input to a spike FFN leads to its output that exhibits massive activations.

Key Takeaway 2: When the MAGC of an input embedding to the spike FFN takes a sufficiently large (or small) value, the output of the FFN asymptotically approaches $\begin{array} { r } { x ^ { 2 } \sum _ { j = 1 } ^ { m } \alpha _ { j } \omega _ { j } } \end{array}$ , where $\omega _ { j }$ is the normalized $j \cdot$ -th column of $W _ { \mathrm { d o w n } }$ of the FFN and $x$ is the value of the MAGC. Moreover, the coefficients $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { m } )$ concentrate on one (or a few) dimensions. As a result, the FFN output is reduced to one normalized column of $W _ { \mathrm { d o w n } }$ or a combination of a few normalized columns of $W _ { \mathrm { d o w n } } .$ , multiplied by their respective coefficient and $x ^ { 2 }$ . Since these columns exhibit the shape of massive activations, the output therefore exhibits massive activations.

## 5 EXTENSION

## 5.1 EXTENSION TO OTHER LLMS

<table><tr><td>Model Name</td><td>Spike FFN Block</td><td>MAGC(i*)</td><td>Direction</td><td>Concentrated Columns(J)</td></tr><tr><td>LLaMA-3.2-1B</td><td>1</td><td>894</td><td>+</td><td>1417</td></tr><tr><td>LLaMA-2-7B</td><td>1</td><td>310</td><td>+</td><td>10411,7890</td></tr><tr><td>Qwen2.5-1.5B</td><td>1</td><td>1395</td><td></td><td>5601, 3295, 3750</td></tr><tr><td>QwQ-32B</td><td>4</td><td>4626</td><td></td><td>11696, 1368</td></tr><tr><td>Ministral-3-3B</td><td>2</td><td>2</td><td></td><td>0,2</td></tr><tr><td>Deepseek-R1-8B</td><td>1</td><td>717</td><td>+</td><td>2427, 198</td></tr></table>

Table 1: MAGC in different models. The spike FFN block, $\mathbf { M A G C } \ i ^ { * }$ , and concentrated column ids are counted from 0. The concentrated columns are identified by first selecting the three columns $\omega _ { j }$ of the down-projection matrix with the largest absolute $\alpha _ { j }$ values and then filtering out the selected columns that do not satisfy $| \cos ( \omega _ { j } , y _ { 0 } ) | \bar { > } 0 . 9$

In this section, we examine the generality of our findings across LLMs of different model families and sizes. Specifically, we study LLaMA-2-7B, Qwen2.5-1.5B, QwQ-32B, Ministral-3-3B-Instruct-2512, and DeepSeek-R1-Distill-Llama-8B. Together with LLaMA-3.2-1B, these models cover four model families and range from 1B to 32B parameters. Based on our empirical findings, we propose an empirical method to identify the MAGC as detailed in Appendix D. For each model, we repeat the procedure in the preceding sections: we localize the spike FFN, identify MAGC i<sup>∗</sup> and its gating direction, and find the dominant column(s) J of $W _ { \mathrm { d o w n } } .$ Table 1 summarizes the results, and the corresponding visualizations are provided in Appendix D.

We highlight the following findings. First, the spike FFN consistently occurs near the beginning of the network. Notably, even for QwQ-32B with 64 transformer blocks, the spike FFN is located at block 4. Second, the position $i ^ { * }$ of MAGC is fixed for a specific model but differs across models, as does the gating direction. Third, the dominant column is not always unique. For LLaMA-3.2-1B, the energy distribution $p _ { j } ( \pmb { \alpha } )$ is dominated by a single column of $\dot { W } _ { \mathrm { d o w n } }$ . For the remaining models, however, two or three columns share the dominant energy. As shown in Appendix $\mathrm { D , }$ the columns in J exhibit highly similar shapes and concentrate their mass on the same small set of output channels, so their weighted combination ${ \pmb y } ^ { * }$ remains concentrated and reproduces the pattern of massive activations. In contrast, a randomly selected column distributes its mass much more broadly across channels.

## 5.2 RELATION TO ATTENTION SINK

Attention sink is commonly attributed to massive activations (Sun et al., 2024; Yan et al., 2024). Our previous results show that massive activations can be controlled by MAGC. We now extend this finding to adjust attention sink using MAGC.

Experimental Setup. We feed ten natural sentences to LLaMA-3.2-1B and intervene only on MAGC of the first token: at the input to the spike FFN (block 1), we overwrite $h _ { 0 } ( i ^ { * } )$ with a prescribed value c and leave all other channels and all other tokens unchanged. We then measure the attention sink of the first token in the following block (block 2) using the sink score (Gu et al., 2025),

$$
s _ { 0 } = \frac { 1 } { H ( T + 1 ) } \sum _ { h = 0 } ^ { H - 1 } \sum _ { t = 0 } ^ { T } A _ { t , 0 } ^ { h } ,\tag{12}
$$

where $A _ { t . 0 } ^ { h }$ denotes the post-softmax attention that the t-th query assigns to the first token in the h-th head, $T + { \dot { 1 } }$ is the sequence length, and H is the number of attention heads. Since the first token is visible to all queries under the causal mask, $s _ { 0 }$ is the average fraction of attention that the first token receives, and $s _ { 0 }$ close to 1 indicates a dominant attention sink.

![](images/bd7ed9bb9ae7b7d4980bfcc9beb6307483e5189aa6fa0a2ee7823195e8d38d56.jpg)

![](images/1dcd856a82f47e4d023649ea4a31fbb9e99ceb0798676ba05f1c9823f3a360ba.jpg)

![](images/fd55084db742c808d3182751aaeadec4356af72160fffacfcdae79e89ae21545.jpg)  
Figure 4: Sink score of the first token in LLaMA-$3 . { \bar { 2 } } – 1 \mathbf { B }$ , measured in the attention of block 2, as a function of the value assigned to MAGC $( i ^ { * } =$ 894) of the first token at the input of the spike FFN (block 1). Thin curves correspond to the ten individual sentences, the thick curve to their mean, and the shaded area to one standard deviation. The unmodified average value of MAGC is 8.81.

Results. Figure 4 reports the sink score $s _ { 0 }$ as a function of MAGC value c. Without intervention, the original MAGC value is 8.81, at which the first token attracts $s _ { 0 } = 0 . 8 9$ of the attention mass. As c varies, the sink score shows a sharp step-like transition. For $c \geq - 2 . 5 , s _ { 0 }$ stays near 0.89, together with massive activations, as shown at $c = + 2 5$ . For sufficiently negative c, the sink collapses. The sink score drops to about 0.46 for $c \leq - 7 . 5$ , while the massive activations disappear, as shown at $c = - 2 5$ . The transition is consistent across sentences, with standard deviation below 0.03 throughout the sweep. Therefore, adjusting the MAGC value in the embedding of an input spike token to the spike FFN can adjust the sink mass on the token.

## 6 CONCLUSION AND FUTURE WORK

In this study, we show how tokens develop massive activations in pretrained transformers and identify the massive activation gating channel (MAGC): a channel in the input to an early FFN whose value gates the emergence of massive activations. We show both empirically and theoretically that once MAGC is large or small enough, the spike FFN asymptotically collapses into a quadratic mixing of a few highly concentrated columns of the down-projection matrix, which reproduces the massiveactivation pattern. This finding holds across six LLMs from four families ranging from 1B to 32B parameters.

Our analysis has several limitations that suggest directions for future work. First, it focuses on decoder-only LLMs with gated (such as SwiGLU) FFNs; extending the framework to mixture-ofexperts architectures is a natural next step. Second, we localize a single spike FFN and a single gating channel per model, whereas a few models exhibit more diffuse behavior across several channels or columns; a multi-channel account may be needed for such cases. Finally, our study is purely mechanistic at inference time and does not explain why a particular channel becomes MAGC during pretraining, which we leave for future investigation.

## AI USE STATEMENT

In this work, we used generative AI tools for paper proofreading and code writing. We have not used generative AI tools for literature review, idea formalization, and drafting the manuscript. We have reviewed all AI-assisted work. For example, LLM-generated code was verified by the authors. We take responsibility for the originality of this work.

## REPRODUCIBILITY STATEMENT

We have released all code needed to reproduce the results in this work at GitHub<sup>1</sup>. It includes scripts for identifying the MAGC, running the intervention experiments, and generating figures. All experiments use publicly available pretrained models.

## REFERENCES

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4895–4901, 2023.

Federico Barbero, Alvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Petar Velickovi ˇ c,´ Razvan Pascanu, and Michael M Bronstein. Why do llms attend to the first token? In Second Conference on Language Modeling, 2025.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In International Conference on Learning Representations, volume 2025, pp. 97114–97144, 2025.

Xu Liu, Guikun Chen, and Wenguan Wang. Sinktrack: Attention sink based context anchoring for large language models. In The Fourteenth International Conference on Learning Representations, 2026.

Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Seungwoo Son, Wonpyo Park, Woohyun Han, Kyuyeun Kim, and Jaeho Lee. Prefixing attention sinks can mitigate activation outliers for large language model quantization. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2242–2252, 2024.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. In First Conference on Language Modeling, 2024.

Shangwen Sun, Alfredo Canziani, Yann LeCun, and Jiachen Zhu. The spike, the sparse and the sink: Anatomy of massive activations and attention sinks. arXiv preprint arXiv:2603.05498, 2026.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Ruiqing Yan, Xingbo Du, Haoyu Deng, Linghan Zheng, Qiuzhuang Sun, Jifang Hu, Yuhang Shao, Penghao Jiang, Jinrong Jiang, and Lian Zhao. Unveiling and controlling anomalous attention distribution in transformers. arXiv preprint arXiv:2407.01601, 2024.

Zhongzhi Yu, Zheng Wang, Yonggan Fu, Huihong Shi, Khalid Shaikh, and Yingyan Celine Lin. Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration. arXiv preprint arXiv:2406.15765, 2024.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Stephen Zhang, Mustafa Khan, and Vardan Papyan. Attention sinks: A’catch, tag, release’mechanism for embeddings. Advances in Neural Information Processing Systems, 38:83140–83181, 2026.

## A GENERALIZATION OF NEAR IDENTITY ASSUMPTION FOR SILU

Following Sun et al. (2026), we compute the cosine similarity between the gate pre-activation $W _ { \mathrm { g a t e } } h _ { t }$ and its activation Si $\mathrm { . U } ( W _ { \mathrm { g a t e } } h _ { t } )$ , and the ratio of their $\ell _ { 2 }$ norms for spike tokens. Using the randomly generated sequences described in Section 3, we report in Table A1 both metrics averaged over the first tokens of these sequences. For SiLU to behave approximately as an identity function, both metrics should be close to 1. Among the models examined, this condition is approximately satisfied only by LLaMA-2-7B.

<table><tr><td>Model</td><td>Metric</td><td>Token 0</td></tr><tr><td>LLaMA-2-7B</td><td>Cosine sim. Norm ratio</td><td>0.9719 0.8103</td></tr><tr><td>LLaMA-3.2-1B</td><td>Cosine sim. Norm ratio</td><td>0.8623 0.6232</td></tr><tr><td>Qwen2.5-1.5B</td><td>Cosine sim. Norm ratio</td><td>0.2276 0.1564</td></tr><tr><td>QwQ-32B</td><td>Cosine sim. Norm ratio</td><td>0.2990 0.1186</td></tr><tr><td>Ministral-3-3B</td><td>Cosine sim. Norm ratio</td><td>0.7895 0.7897</td></tr><tr><td>Deepseek-R1-8B</td><td>Cosine sim. Norm ratio</td><td>0.9215 0.7720</td></tr></table>

Table A1: Input-output characteristics of SiLU in the spike FFN across models.

![](images/8eb145a0c9a89a5d30b6bebdcd22348178a709e9014c919377e4541f64c0ee3a.jpg)

![](images/dd5143e940a4f80985d0455cdb905d1d9d5bd9f30958456b3a568078195ae6ce.jpg)  
Figure A1: Top-3 channel magnitudes of transformer block outputs $x + u + y$ in Llama-3.2-1B and their differences.

![](images/3805d99efca0a1769a52703cf13d0b1ff9224b54b518683958fb488f31be1b3c.jpg)  
Figure A2: Inputs and Outputs of the Spike FFN.

## B ADDITIONAL EXPERIMENTS FOR LLAMA-3.2-1B

## B.1 MASSIVE ACTIVATIONS

Figure A1 shows the block-wise differences of the top-3 channel magnitudes of the transformer block outputs ${ \pmb x } + { \pmb u } + { \pmb y }$ in LLaMA-3.2-1B. As depicted, the differences exhibit a single sharp jump at the second transformer block, where the spike FFN triggers massive activations. In the subsequent middle blocks, the differences remain close to zero, indicating that no further massive activations are triggered and that the residual stream alone preserves the existing ones until the late blocks.

## B.2 INPUT AND OUTPUT OF THE SPIKE FFN FOR OTHER SAMPLE SEQUENCES

Following Section 3, we depict the activations of the other 4 randomly generated sequences in Figure A2.

## B.3 PARETO INPUTS FOR THE SPIKE FFN

![](images/51421acb61ed83da7a648186c72e0ea4a4610218da034e822301407349c8b9a2.jpg)

![](images/3061cae4194125fef96cfd5d7bed72b9fc536094f5f081be237d64a1cb6e6abf.jpg)

![](images/6eb2829b01ca521cc5dc482cd5d49179e2012b632770a2f466f49bc9b9875292.jpg)

![](images/09ab04eb781e576bfea96a53642733620a16a3f9f842bbcdfc9b10df7fdf8811.jpg)

![](images/ec88ff52cec97ddc5cf3ee1180e5d5ae3151fd1db2b14c1ea4439b01b7b7d477.jpg)

![](images/c9af234c5a4cbc3046acf26a6e8b5c0f7d4ab672d6890fc97d679bc2a1240580.jpg)  
Figure A3: From top to bottom, the figure shows the output of the spike FFN when the input vector values are randomly sampled from a Pareto distribution $( a = 3 )$ , with the input value of the 894th (MAGC), 0th, or 1742nd channel swapped with the maximum input value, respectively.

![](images/e3d3f3f6917f1e23910455f64a83b954a512c924dd022e04693c8ca138129c5c.jpg)  
Input Pareto(a=5), peak@ch 0

Block 1 FFN output (peak@ch 894, a=5)  
![](images/a9112216d52aa04bc772d1750df8a1fef33fb904a30f73c5e13b2374d2380c12.jpg)  
Block 1 FFN output (peak@ch 0, a=5)

![](images/c1c4eadea3aa8bc13457770aad9c463bba6ccdf07cd7a8dd7ddf721f3f6e2f3a.jpg)

![](images/efacf990f4046906b454467783f7e99cc3cd96e6c30eb03fa6f81b6631b8a352.jpg)  
Block 1 FFN output (peak@ch 1742, a=5)

Input Pareto(a=5), peak@ch 1742  
![](images/bb6bc60077f311c63e85accb6820ec545e95a61bfc8d9f356e43dde7a575f8ca.jpg)

![](images/08ce9adc1d7371433d73fd75156ab7500b89f9a48bc3b0e13d9953ccbffd9d09.jpg)  
Figure A4: From top to bottom, the figure shows the output of the spike FFN when the input vector values are randomly sampled from a Pareto distribution $( a = 5 )$ , with the input value of the 894th (MAGC), 0th, or 1742nd channel swapped with the maximum input value, respectively.

## C PROOFS

## C.1 PROOF OF THEOREM 4.1

Theorem C.1. Given input $\pmb { h } ( x ) = x \cdot \pmb { e } _ { i } + \pmb { \bar { h } }$ . When $x \to + \infty ,$ , the output of an FFN has a quadratic form $\pmb { y } ( \pmb { h } ( x ) ) = x ^ { 2 } \cdot \pmb { y } ^ { * } + o ( \dot { x } ^ { 2 } )$ with constant $\boldsymbol { y } ^ { * } \in \mathbb { R } ^ { d }$ equals to

$$
{ \pmb y } ^ { * } = \sum _ { j = 1 } ^ { m } { \boldsymbol \alpha } _ { j } \cdot { \boldsymbol \omega } _ { j } ,\tag{A1}
$$

where $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { m } )$ is the combination of coefficients in the weight matrices:

$$
\alpha _ { j } = \left\{ \begin{array} { l l } { W _ { \mathrm { g a t e } } ^ { ( j , i ) } W _ { \mathrm { u p } } ^ { ( j , i ) } \left. W _ { \mathrm { d o w n } } ^ { ( : , j ) } \right. _ { 2 } } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } > 0 } \\ { 0 } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } \le 0 } \end{array} . \right.\tag{A2}
$$

Proof. Based on Equation 4, the linear representation of $W _ { \mathrm { g a t e } } h ( x )$ and $W _ { \mathrm { u p } } h ( x )$ can be written as

$$
W _ { \mathrm { g a t e } } h ( x ) = x W _ { \mathrm { g a t e } } ^ { ( \therefore , i ) } + b _ { \mathrm { g a t e } }\tag{A3}
$$

$$
W _ { \mathrm { u p } } h ( x ) = x W _ { \mathrm { u p } } ^ { ( \cdot , i ) } + b _ { \mathrm { u p } }\tag{A4}
$$

where $\begin{array} { r } { \pmb { b } _ { \mathrm { g a t e } } = \sum _ { i = 1 } ^ { d } \bar { h } _ { i } \pmb { W } _ { \mathrm { g a t e } } ^ { ( : , i ) } } \end{array}$ and $\begin{array} { r } { \boldsymbol { b } _ { \mathrm { u p } } = \sum _ { i = 1 } ^ { d } \bar { h } _ { i } \boldsymbol { W } _ { \mathrm { u p } } ^ { ( : , i ) } } \end{array}$ are both constant under x.

We have the asymptotic property of SiLU(x):

$$
\operatorname* { l i m } _ { x \to + \infty } \frac { \mathrm { S i L U } ( W _ { \mathrm { g a t e } } h ( x ) ) } { x } = \left\{ W _ { \mathrm { g a t e } } ^ { ( j , i ) } \quad W _ { \mathrm { g a t e } } ^ { ( j , i ) } \geq 0 \right.\tag{A5}
$$

and the asymptotic property of $W _ { \mathrm { u p } } h ( x )$

$$
\operatorname* { l i m } _ { x \to + \infty } { \frac { W _ { \mathrm { u p } } h ( x ) } { x } } = W _ { \mathrm { u p } } ^ { ( : , i ) }\tag{A6}
$$

Finally,

$$
\operatorname* { l i m } _ { x \to + \infty } \frac { y ( h ( x ) ) } { x ^ { 2 } } = W _ { \mathrm { d o w n } } \operatorname* { l i m } _ { x \to + \infty } \frac { \mathrm { S i L U } ( W _ { \mathrm { g a t e } } h ( x ) ) } { x } \odot \operatorname* { l i m } _ { x \to + \infty } \frac { W _ { \mathrm { u p } } h ( x ) } { x } = \sum _ { j = 1 } ^ { m } \alpha _ { j } \cdot \omega _ { j } ,\tag{A7}
$$

where $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { m } )$ is

$$
\alpha _ { j } = \left\{ \begin{array} { l l } { W _ { \mathrm { g a t e } } ^ { ( j , i ) } W _ { \mathrm { u p } } ^ { ( j , i ) } \left. W _ { \mathrm { d o w n } } ^ { ( : , j ) } \right. _ { 2 } } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } > 0 } \\ { 0 } & { W _ { \mathrm { g a t e } } ^ { ( j , i ) } \le 0 } \end{array} . \right.\tag{A8}
$$

## C.2 PROOF OF PROPOSITION 4.6

Recall that

$$
\pmb { y } ^ { * } = \sum _ { j = 1 } ^ { m } \alpha _ { j } \omega _ { j } , \qquad \| \omega _ { j } \| _ { 2 } = 1 .
$$

The energy distribution of α is

$$
p _ { j } ( \pmb { \alpha } ) = \frac { \alpha _ { j } ^ { 2 } } { \lVert \pmb { \alpha } \rVert _ { 2 } ^ { 2 } } , \qquad j = 1 , \dotsc , m .
$$

Let $j ^ { * } = \arg \operatorname* { m a x } _ { j } p _ { j } ( \pmb { \alpha } )$ be the dominant coefficient index. Then ${ \pmb y } ^ { * }$ can be decomposed into the dominant column and the residual part:

$$
\pmb { y } ^ { * } = \alpha _ { j ^ { * } } \omega _ { j ^ { * } } + \sum _ { j \neq j ^ { * } } \alpha _ { j } \omega _ { j } .
$$

Define the residual vector

$$
r = \sum _ { j \neq j ^ { * } } \alpha _ { j } \omega _ { j } .
$$

The residual amplification factor is defined as in Definition 4.5,

$$
\kappa = \frac { \| \pmb { r } \| _ { 2 } } { \sqrt { \sum _ { j \neq j ^ { \ast } } \alpha _ { j } ^ { 2 } } } .
$$

The angular deviation between the true asymptotic output and the dominant column is

$$
| \sin \angle ( \pmb { y } ^ { * } , \pmb { \omega } _ { j ^ { * } } ) | = \sqrt { 1 - \left( \frac { \pmb { y } ^ { * } \cdot \pmb { \omega } _ { j ^ { * } } } { \| \pmb { y } ^ { * } \| _ { 2 } \| \pmb { \omega } _ { j ^ { * } } \| _ { 2 } } \right) ^ { 2 } } .
$$

Finally, for $\tau \in ( 0 , 1 - \frac { 1 } { d } )$ , define

$$
\Delta ( \tau ) = - \tau \log \tau - ( 1 - \tau ) \log ( 1 - \tau ) + \tau \log ( d - 1 ) .
$$

Lemma C.2. Assume | sin $\begin{array} { r } { \angle ( { \pmb y } ^ { \ast } , { \pmb \omega } _ { j ^ { \ast } } ) | \leq 1 - \frac { 1 } { d } } \end{array}$ . Then,

$$
| R ( \pmb { y } ^ { \ast } ) - R ( \omega _ { j ^ { \ast } } ) | \leq \Delta \left( | \sin \angle ( \pmb { y } ^ { \ast } , \pmb { \omega } _ { j ^ { \ast } } ) | \right) .
$$

Proof. Based on the Fannes-Audenaert inequality:

$$
| R ( \pmb { y } ^ { * } ) - R ( \omega _ { j ^ { * } } ) | = | H ( \omega _ { j ^ { * } } ) - H ( \omega _ { j ^ { * } } ) | \leq \Delta ( t ) ,
$$

where $\begin{array} { r } { t : = \frac { 1 } { 2 } \| p ( \pmb { y } ^ { * } ) - p ( \pmb { \omega } _ { j ^ { * } } ) \| } \end{array}$ <sub>1</sub> and H is the entropy function. We have

$$
\begin{array} { r l } { \frac { 1 } { 2 } | | \boldsymbol { p } ( \boldsymbol { y } ^ { \prime } ) - \boldsymbol { p } ( \omega _ { \mathcal { S } ^ { \prime } } ) | | = \frac { 1 } { 2 } \frac { \delta } { 1 \omega ^ { 2 } } | ( \frac { \boldsymbol { y } ^ { \mathcal { S } } } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } ) ^ { 2 } - ( \frac { \omega _ { \mathcal { S } ^ { \prime } } } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } ) ^ { 2 } | } & { } \\ { = } & { \frac { 1 } { 2 } \frac { \delta } { 1 \omega ^ { 2 } } | ( \frac { \boldsymbol { y } ^ { \mathcal { S } } } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } - \frac { \omega _ { \mathcal { S } ^ { \prime } } } { \| \omega _ { \mathcal { S } ^ { \prime } } \| _ { 2 } } ) ( \frac { \boldsymbol { y } ^ { \mathcal { S } } } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } + \frac { \omega _ { \mathcal { S } ^ { \prime } } } { \| \omega _ { \mathcal { S } ^ { \prime } } \| _ { 2 } } ) | } \\ & { \leq \frac { 1 } { 2 } [ \frac { \delta } { 1 \omega ^ { 2 } } ( \frac { \delta } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } - \frac { \omega _ { \mathcal { S } ^ { \prime } } } { \| \boldsymbol { y } \omega _ { \mathcal { S } ^ { \prime } } \| _ { 2 } } ) ] ^ { - 1 / 2 } [ \frac { \delta } { 1 - 1 } ( \frac { \boldsymbol { y } ^ { \prime } } { \| \boldsymbol { y } ^ { \prime } \| _ { 2 } } + \frac { \omega _ { \mathcal { S } ^ { \prime } } } { \| \boldsymbol { y } \omega _ { \mathcal { S } ^ { \prime } } \| _ { 2 } } ) ^ { 2 } ] ^ { 1 / 2 } } \\ &  = \frac { 1 } { 2 } [ 2 - 2 \ \end{array}
$$

Since $\Delta ( t )$ is increasing on $\begin{array} { r } { 0 \leq t \leq 1 - \frac { 1 } { d } } \end{array}$

$$
| R ( \pmb { y } ^ { * } ) - R ( \omega _ { j ^ { * } } ) | \leq \Delta \left( \operatorname* { m i n } \left\{ | \sin \angle ( \pmb { y } ^ { * } , \omega _ { j ^ { * } } ) | , 1 - \frac { 1 } { d } \right\} \right) = \Delta \left( | \sin \angle ( \pmb { y } ^ { * } , \omega _ { j ^ { * } } ) | \right) .
$$

Proposition C.3. Let

$$
\tau = \kappa \sqrt { \frac { 1 - p _ { j ^ { * } } ( \alpha ) } { p _ { j ^ { * } } ( \alpha ) } } .\tag{A9}
$$

We have

$$
| R ( \pmb { y } ^ { \ast } ) - R ( \omega _ { j ^ { \ast } } ) | \leq \Delta \left( \tau \right) ,\tag{A10}
$$

where $\begin{array} { r } { f o r \tau \in \bigl ( 0 , 1 - \frac { 1 } { d } \bigr ) , \Delta ( \tau ) = - \tau \log \tau - ( 1 - \tau ) \log ( 1 - \tau ) + \tau \log ( d - 1 ) . } \end{array}$

Proof. Recall $\begin{array} { r } { r = \sum _ { j \neq j ^ { * } } \alpha _ { j } \omega _ { j } } \end{array}$ , then $\pmb { y } ^ { * } = \alpha _ { j ^ { * } } \omega _ { j ^ { * } } + \pmb { r }$ . Since $\| \omega _ { j ^ { * } } \| _ { 2 } = 1$ , decompose the residual vector as

$$
r = r _ { \parallel } \omega _ { j ^ { * } } + { r } _ { \perp } , \qquad r _ { \perp } \perp \omega _ { j ^ { * } } .
$$

We have

$$
\begin{array} { r } { \pmb { y } ^ { \ast } = ( \alpha _ { j ^ { \ast } } + r _ { \parallel } ) \pmb { \omega } _ { j ^ { \ast } } + \pmb { r } _ { \perp } . } \end{array}
$$

The angular deviation from $\omega _ { j ^ { \ast } }$ is caused only by the perpendicular residual component. Hence,

$$
\begin{array} { r l } & { | \sin \angle ( \pmb { y } ^ { * } , \pmb { \omega } _ { j ^ { * } } ) | = \frac { \| \pmb { r } _ { \bot } \| _ { 2 } } { \big [ ( \alpha _ { j ^ { * } } + r _ { \parallel } ) ^ { 2 } + \| \pmb { r } _ { \bot } \| _ { 2 } ^ { 2 } \big ] ^ { 1 / 2 } } } \\ & { \qquad \le \frac { \sqrt { r _ { \parallel } ^ { 2 } + \| \pmb { r } _ { \bot } \| _ { 2 } ^ { 2 } } } { | \alpha _ { j ^ { * } } | } = \frac { \| \pmb { r } \| _ { 2 } } { | \alpha _ { j ^ { * } } | } . } \end{array}
$$

The above inequality follows because, after squaring and rearranging, it is equivalent to

$$
\begin{array} { r } { 0 \leq \left( r _ { \parallel } ( \alpha _ { j ^ { * } } + r _ { \parallel } ) + \| r _ { \perp } \| _ { 2 } ^ { 2 } \right) ^ { 2 } . } \end{array}
$$

Moreover,

$$
\frac { \| r \| _ { 2 } } { | \alpha _ { j ^ { * } } | } = \kappa \frac { \sqrt { \sum _ { j \neq j ^ { * } } \alpha _ { j } ^ { 2 } } } { | \alpha _ { j ^ { * } } | } = \kappa \sqrt { \frac { 1 - p _ { j ^ { * } } ( \alpha ) } { p _ { j ^ { * } } ( \alpha ) } } .
$$

Therefore,

$$
| \sin \angle ( { \pmb y } ^ { \ast } , { \pmb \omega } _ { j \ast } ) | \leq \kappa \sqrt { \frac { 1 - p _ { j ^ { \ast } } ( { \pmb \alpha } ) } { p _ { j ^ { \ast } } ( { \pmb \alpha } ) } } .
$$

Based on Lemma C.2, it is sufficient to prove Proposition 4.6.

## D ADDITIONAL RESULTS

Identification of the Spike FFN and MAGC. We identify the spike FFN and MAGC by comparing the first (spike) token and the second (non-spike) token of the randomly generated sequences. Let y<sup>l</sup> denote the average FFN output of the first token in the l-th transformer block. We regard an FFN as the spike FFN if its output exhibits an outlier channel (Sun et al., 2024), i.e.,

$$
\operatorname* { m a x } _ { j } \left| y _ { 0 } ^ { l } ( j ) \right| > 1 0 0 \quad \mathrm { a n d } \quad \operatorname* { m a x } _ { j } \left| y _ { 0 } ^ { l } ( j ) \right| > 1 0 \cdot \operatorname* { m e d i a n } _ { j } \left| y _ { 0 } ^ { l } ( j ) \right| .\tag{A11}
$$

Given the spike FFN, we then identify MAGC from its inputs. Let $h _ { 0 }$ and $\pmb { h } _ { 1 }$ be the average inputs to the spike FFN for the first and second tokens, respectively. The gating channel should be the channel where the two inputs deviate the most. We therefore locate MAGC as

$$
i ^ { * } = \arg \operatorname* { m a x } _ { j } \left| h _ { 0 } ( j ) - h _ { 1 } ( j ) \right| ,\tag{A12}
$$

and determine the gating direction by the sign of $h _ { 0 } ( i ^ { * } ) - h _ { 1 } ( i ^ { * } )$ : a positive (negative) sign indicates that massive activations emerge when the value of MAGC becomes sufficiently large (small).

This appendix reports the detailed results for the models summarized in Table 1: LLaMA-2-7B (Figure A5), Qwen2.5-1.5B (Figure A6), QwQ-32B (Figure A7), Ministral-3-3B-Instruct-2512 (Figure A8), and DeepSeek-R1-Distill-Llama-8B (Figure A9). For each model, we present (i) the input h and outputs y of the spike FFN for a random input sequence, (ii) the coefficient vector α defined in Theorem 4.1, where for models with a negative gating direction α is computed under the limit $x \to - \infty$ and the condition on $W _ { \mathrm { g a t e } } ^ { ( j , i ) }$ flips sign accordingly, and (iii) the dominant columns (the columns most aligned with the output y<sub>0</sub>, as reported in Table 1) together with a randomly selected column for comparison.

![](images/aa9d14e9a5fcdb29da6bc98a2f08d2901a6d66a6c51f8b86224ab09d7cfb073a.jpg)

![](images/67682e636dfcaeab59eeb1d1c61cbeefd4c4c4fdeed9b4ec6785030fb5574cf8.jpg)

![](images/4be4e27ab7140a2428ade1d5b07a95f428eac491cadcfe3fc9d65832275f4626.jpg)

![](images/c05e2ded065983b989a58baab5af742c3e04d215dd1128faf421430f2853d6ca.jpg)

![](images/6484c969b795cfe70c094ca664571768e454389730913e061c0bbba79b188922.jpg)  
(e) ω<sub>7890</sub>.

![](images/a1e2042bdbccc6faa89bdebbdbbc4459bfd745c23173b246c325b1adab47132a.jpg)  
(f) ω<sub>0</sub>.  
Figure A5: Results for LLaMA-2-7b-hf.

![](images/9f03c5b17af60d434286565f0324b23bde2a049ac16d166f78671c30175fb70d.jpg)

![](images/582a6868cd51c2c9e06ac7d53d7544edfc807386a9290b2aa0af63f6b8359485.jpg)  
(b) Output y of FFN.

![](images/d1bcca622254b1b336249417c7ad5f700705b4fec2412ad535ea08b63bbeaa15.jpg)  
(c) α vector when i<sup>∗</sup> = 1395.

![](images/997d3fb84ed1f8858fcb7e57080f3c1bc94b91313ca1afa7c09e75cdc385977c.jpg)

![](images/78580ef34907240ac61a9a2397a7d3e3611db90fb498e0992f27a309f5c473c4.jpg)  
(e) ω<sub>3295</sub>.

(d) ω<sub>5601</sub>.  
![](images/f832abcbbb41a70afa58b5f3940d850226768a7cc59899c0b91f5b16d0c58dab.jpg)  
(f) ω<sub>3750</sub>.  
Figure A6: Results for Qwen2.5-1.5b.

![](images/b7738dfa1e638b57c8620fbce98f82fe81167452a9369e0d3d4a51316badc51e.jpg)

![](images/3173a09f6638fa8da9374d8d37dc13e1285ed7a41d65b4988181a26fc6c37530.jpg)

![](images/397136f3fe575b6ff456d79ecf983a1b0b71c41f6d56384337e7e2ca42c5e0b7.jpg)  
(c) α vector when $i ^ { * } = 4 6 2 6$

(b) Output y of FFN.  
![](images/11f27c39f5f031646a38ea1f16b9576b1d6faaac53610753fa109f14ae731921.jpg)

![](images/aad2121fcdd437f389faf25870c0376813c09b5fa8affc2705bf86a38c6af769.jpg)  
(e) ω<sub>1368</sub>.

(d) ω<sub>11696</sub>.  
![](images/fa46c0c2773f7c09ffadc87535ab65f8e399776b1e06c039ca604c1c530a092f.jpg)  
(f) ω<sub>0</sub>.  
Figure A7: Results for QwQ-32B. For QwQ-32B, several of the channels with the largest input difference $| h _ { 0 } ( j ) - h _ { 1 } ( j ) |$ coincide with massive-activation channels propagated through the residual stream; we therefore identify the MAGC $( i ^ { * } = 4 6 2 6$ , negative gating direction) by traversing the ten channels with the largest such difference following Equation A12 and selecting the first one that is not itself a massive-activation channel, rather than simply taking the single largest.

![](images/d1ac3ea6c18712ae689c2f91fbaa8b46d55e160505a5545c8a0ac60ec52dcb8a.jpg)

![](images/7fe5d032d1666f8d73ef93f6f0355b25ead9b0a9b75dc3233b8db34e173e47e8.jpg)

![](images/fa9162737ee96825fe1e6cbdc2f25dc4038b522d4647b9993ed851abc12fe153.jpg)  
(c) α vector when i<sup>∗</sup> = 2.

![](images/498397eaee4c638cda19600341a72f25bf5f389de28fdd254a26636f63a2e3ae.jpg)

![](images/d6f4403275f746a7c8dc0156e4445e8b467047986fa090b6f87b0f3224751b82.jpg)  
(e) ω<sub>2</sub>.

![](images/cc5726560ea9ca4f9e18176ba09e47a251d6988c5e4a543a185da107a0a335f6.jpg)  
(f) ω<sub>1</sub>.  
Figure A8: Results for Ministral-3-3B-Instruct-2512. Note that columns 0 and 2 belong to $j ^ { * }$ . Instead of choosing 0 as the random column for comparison, we select column 1.

![](images/499ed78e8771e9f838caf9a75bcaa6a3c1687813ee8a7792165e6cdfa26764f1.jpg)  
(a) Input h to FFN.

![](images/993272b5f0e2fedaeeff1043a7143a119f525745950f362dea913a1bf5c54ad4.jpg)  
(b) Output y of FFN.

![](images/946e19dd0ad30d3f394dc4c303559571facc1e3395713f023c7011673474697a.jpg)  
(c) α vector when i<sup>∗</sup> = 717.

![](images/c6db14fa59d4a3968a54910bb203d708a6eeb3478793ed56e897fd241393b0b4.jpg)  
(d) ω<sub>2427</sub>.

![](images/a5ecfc0ca33146ca2cf66ab82f38e9530671869f1f0a20f238962b93db79559e.jpg)  
(e) $\omega _ { 1 9 8 }$

![](images/dc5f160e6c5596ff82ce48d57b61772abfb74b3a7525a7b206390db68c22f3c6.jpg)  
(f) ω<sub>0</sub>.  
Figure A9: Results for DeepSeek-R1-Distill-Llama-8B.