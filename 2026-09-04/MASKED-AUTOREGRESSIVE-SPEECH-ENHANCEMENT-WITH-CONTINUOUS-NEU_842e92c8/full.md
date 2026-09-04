# MASKED AUTOREGRESSIVE SPEECH ENHANCEMENT WITH CONTINUOUS NEURAL AUDIO CODEC REPRESENTATIONS

Yoto Fujita<sup>1,2</sup> Simon Leglaive<sup>1</sup> Laurent Girin<sup>2</sup>

<sup>1</sup> CentraleSupelec, IETR (UMR CNRS 6164), France´ <sup>2</sup> Univ. Grenoble Alpes, CNRS, Grenoble-INP, GIPSA-lab, France

## ABSTRACT

Most previous work on speech enhancement (SE) based on masked generative modeling relied on discrete token representations of audio signals, obtained using neural audio codecs (NACs). However, a recent study has shown that continuous latent representations of NACs can be advantageous for SE in terms of speech quality and intelligibility. In this work, we propose masked autoregressive SE (MARSE), a method for SE based on iterative decoding of masked clean speech frames using continuous NAC representations of speech. In particular, we investigate a set of different decoding policies, ceteris paribus, that is, using the same DNN (a Conformer model), the same NAC (the DAC codec) and the same training setup. The results show that MARSE enables a flexible trade-off between SE performance and computational cost. Audio examples and code are available online.<sup>1</sup>

Index Terms— Speech enhancement, masked generative modeling, neural audio codec, autoregressive modeling, speech representations.

## 1. INTRODUCTION

Speech enhancement (SE) in noise has a long history of conventional signal processing techniques [1, 2] and has been largely revisited in the last decade with deep neural networks (DNNs) [3]. DNN-based SE has been increasingly performed in learned latent spaces instead of using conventional representations such as the short-time Fourier transform. In particular, the compact yet expressive discrete latent representations provided by neural audio codecs (NAC), which consist of sequences of vector quantizer codes called tokens, have recently facilitated SE based on token sequence modeling with Transformers [4], in the line of speech language models [5]. The tokens corresponding to a clean speech signal frame at time t are decoded (i.e., predicted) using the noisy signal and possibly previously decoded speech tokens. Token-based SE has been explored with different decoding strategies: non-autoregressive decoding [6, 7], causal autoregressive decoding [6, 8, 9], or non-causal masked generative decoding [10, 11, 12, 13].

This paper aims at both investigating SE models that operate on continuous NAC representations and extending the study of decoding policies for SE. This is motivated by the following three points. First, although token-based SE methods are reported to exhibit good speech quality, they can suffer from limited preservation of acoustic details that matter for speech quality and intelligibility, leading to reduced performance on some downstream tasks such as automatic speech recognition (ASR) [10]. Second, it has been recently shown in [6] that using a continuous NAC representation—that is, embedding vectors at the output of the NAC encoder and before the quantizer—can substantially improve speech intelligibility and quality. Third, most of the token-based methods mentioned above focus on a specific decoding policy with a specific experimental setup, and the trade-off between performance and computational cost regarding decoding policy has not been explored.

![](images/77131a6a689db480b5b17e0725f51a54debd41de4d0bd76645dbc99afa18b541.jpg)  
Fig. 1. Overview of the proposed MARSE method applied on a continuous NAC representation (left: inference; right: training).

Independently of the SE problem, a unified framework called masked autoregressive (MAR) modeling was recently proposed for image generation in [14]. There, decoding is formulated as an iterative unmasking process, providing a unified view of autoregressive (AR) and masked generation. This framework was then adapted for SE with continuous representations in [15]. In the present paper, we expand the work of [6], [14], and [15] and propose masked autoregressive speech enhancement (MARSE), in which iterative decoding is formulated as an AR probabilistic process over blocks of continuous NAC representations. Our objective is not to introduce a new decoding approach, but rather to provide a unified framework encompassing AR, non-AR and masked AR decoding policies. With a particular focus on the role and impact of decoding policies, our work offers complementary insights to previous work. Concretely, we compare several decoding policies as variants of the number of iterations and/or the definition of the blocks. We study their effect on speech quality, intelligibility, and computational cost, by a fair comparison using the same network architecture (a Conformer-based model [16]), the same NAC (Descript Audio Codec [17]), and the same training setup. Experiments show that the proposed MARSE method allows for a flexible trade-off between SE performance and computational cost.

## 2. METHOD

In a probabilistic framework, SE can be defined as the problem of estimating a conditional distribution $p _ { \theta } ( \mathbf { x } | \mathbf { y } )$ of parameters θ, where $\mathbf { x } , \mathbf { y } \in { \mathcal { A } }$ denote the clean and noisy speech signals respectively, defined in some representation domain A. Inspired by [6], we consider a model that operates on the continuous latent representation of a pretrained NAC, before quantization. We thus have $\mathbf { x } = \{ \mathbf { x } _ { t } \in$ $\mathbb { R } ^ { D } \dot  \} _ { t = 1 } ^ { T }$ and $\mathbf { y } = \{ \mathbf { y } _ { t } \in \bar { \mathbb { R } ^ { D } } \} _ { t = 1 } ^ { T }$ , where $T$ denotes the number of time frames and D the NAC’s latent space dimension. Deep learning-based SE methods rely on DNNs to parametrize the distribution $p _ { \theta } ( \mathbf { x } | \mathbf { y } )$ . In a supervised setting, the parameters θ are learned using a labeled dataset of paired clean and noisy speech signals, typically by minimizing the negative log-likelihood $- \ln p _ { \theta } ( \textbf { x } \ | \textbf { y } )$ averaged over the training examples.

## 2.1. Masked autoregressive speech enhancement

Within this probabilistic framework, defining $p _ { \theta } ( \mathbf { x } \mid \mathbf { y } )$ involves (i) defining the distribution of the clean speech variables $\mathbf { x } _ { t } .$ $t ~ \in ~ \{ 1 , . . . , T \}$ , and their temporal dependencies; and (ii) defining the DNN architecture that implements this probabilistic model. In this work, we apply the MAR framework of [14] to the SE problem, resulting in the class of MARSE models, which allows for arbitrary assumptions about these dependencies and tackles SE as an iterative decoding process, as illustrated in the left part of $\operatorname { F i g . } 1 .$ In MARSE, the conditional distribution of x given y is defined $\mathsf { b y } \colon ^ { 2 }$

$$
p _ { \theta } ( \mathbf { x } \mid \mathbf { y } ) = \prod _ { i = 1 } ^ { N } p _ { \theta } \left( \mathbf { x } _ { M ( i ) } \mid \mathbf { y } , \mathbf { x } _ { V ( i ) } \right)\tag{1}
$$

$$
\begin{array} { r l } { \mathbf { \eta } } & { = \prod _ { i = 1 } ^ { N } \prod _ { t \in M ( i ) } \mathcal { N } ( \mathbf { x } _ { t } ; f _ { \theta , t } ( \mathbf { y } , \mathbf { x } _ { V ( i ) } ) , \mathbf { I } ) , } \end{array}\tag{2}
$$

where $N \in \{ 1 , . . . , T \}$ is the number of iterations in the decoding process and

$\mathbf { x } _ { M ( i ) }$ denotes the set of predicted frames at iteration i, with $M ( i ) \subseteq \{ 1 , . . . , T \}$ and such that $\bigcup _ { i = 1 } ^ { N } M ( i ) = \{ 1 , . . . , T \}$ ;

$\mathbf { x } _ { V ( i ) }$ denotes the set of visible (i.e. previously decoded) frames at the beginning of iteration i, with $\bar { V ( 1 ) } = \emptyset , \bar { V ( i + 1 ) } = V ( i )$ ∪ $M ( i )$ and $\bar { M } ( i ) \cap V ( i ) = \emptyset$ for $i \geq 1$

• and $f _ { \theta } : \mathbb { R } ^ { D \times T } \times \mathbb { R } ^ { D \times T } \mapsto \mathbb { R } ^ { D \times T }$ denotes a neural network that processes the noisy speech y and visible clean speech frames $\mathbf { x } _ { V ( i ) }$ to predict $\mathbf { x } _ { M ( i ) }$ . Inspired by [6], we use a Conformer-based network taking as input the temporal concatenation of noisy and partially-masked clean speech representations. The masked clean speech frames at positions $t \in \{ \bar { 1 } , . . . , T \} \setminus V ( i )$ are here replaced by a learnable mask vector.

By construction, the collection of index sets $\{ M ( i ) \} _ { i = 1 } ^ { N }$ is a disjoint partition of $\{ 1 , . . . , T \}$ . The factorization in (1) is therefore an application of the chain rule of probability that constitutes an autoregressive model over blocks offrames, which forms a proper probability density function (PDF) over x as long as each conditional factor in (1) is also a valid PDF. Eq. (2) shows that within a block, the vectors $\mathbf { x } _ { t }$ are assumed to be mutually conditionally independent.

## 2.2. Decoding policies for inference

In MARSE, a fundamental aspect of the model is the definition of the collection of index sets $\{ M ( \boldsymbol { i } ) \} _ { i = 1 } ^ { N }$ , which will be referred to as the decoding policy in the following sections. This policy directly relates to the assumed temporal dependencies between the clean speech frames. The flexibility of MARSE comes from the possibility of designing various decoding policies and, in particular, choosing an arbitrary number of iterations $N \in \{ 1 , . . . , \bar { T } \}$ . Below, we present causal and non-causal decoding policies, which can all be characterized in two steps: (i) defining the number of frames decoded in parallel at iteration i, which is equal to the cardinality of $M ( i )$ in (1), denoted by card(M(i)); (ii) defining the indices of the frames to predict, that is, specifying the set $M ( i )$ given its cardinality. In this work, following [18], the cardinality of $M ( i )$ is fixed according to a predefined cosine schedule. More precisely, card $( M ( i ) ) =$ $n _ { i + 1 } - n _ { i }$ frames, where $\begin{array} { r } { n _ { i } = \mathrm { c a r d } ( V ( i \bar { ) } ) = \left\lfloor T \left( 1 - \dot { \gamma } ( \frac { i - 1 } { N } ) \right) \right\rfloor } \end{array}$ with $\gamma : r \in [ 0 , 1 ] \mapsto \cos \left( { \frac { \pi } { 2 } } r \right) \in [ 0 , 1 ]$ . Here, n<sub>i</sub> denotes the number of predicted frames up to, and not including, iteration $i ,$ such that $n _ { 1 } = \mathrm { c a r d } ( V ( 1 ) ) = 0$ . Then, variants of the decoding policy are obtained by specifying different strategies for defining M(i) at each iteration and/or by choosing a different number of iterations.

Causal decoding. In the block-wise causal AR decoding policy, the set of indices to predict is defined by $M ( i ) = \{ n _ { i } + 1 , n _ { i } +$ $2 , . . . , n _ { i + 1 } \}$ Thus, each block of frames consists of a subset of consecutive frames, and the blocks follow a causal ordering along the iterations. In the limit case where $N = 1$ , we have $M ( 1 ) =$ $\{ 1 , . . . , T \}$ and $V ( 1 ) = \emptyset$ , i.e. all frames are decoded all at once and the decoding is not really AR anymore (we thus can refer to it as non-AR). As can be seen from $( 2 )$ , this non-AR decoding assumes that all variables x<sub>t</sub>, $t \in \{ 1 , . . . , T \}$ , are mutually independent. It constitutes the most naive temporal model. In the limit case where $N = T$ we recover the conventional frame-wise causal AR decoding with $\begin{array} { r } { M ( i ) \ = \ \{ i \} } \end{array}$ and $V ( i ) ~ = ~ \{ 1 , . . . , i - 1 \}$ for all $i \in \{ 1 , . . . , T \}$ This policy is the most expressive in terms of temporal modeling, as no conditional independence assumptions are made. However, it requires $T$ forward passes in the neural network $f _ { \theta } ,$ which can be computationally expensive.

Non-causal decoding with oracle index selection. Contrary to the previous policy, in the non-causal decoding policy a masked frame at a given time step can be decoded conditioned on frames in the future. The intuition behind a non-causal policy for SE is that frames associated with high signal-to-noise ratio can be decoded first, and then used to decode noisier frames from the past. In the non-causal decoding policy with oracle index selection, at iteration i, we first decode all the frames that remain to be decoded. Then, we use the ground-truth clean speech to select, among those frames, the $\mathrm { c a r d } ( M ( i ) )$ ) ones that have the lowest prediction error (in terms of squared error). This decoding policy is not realistic, as it requires the availability of the ground-truth clean speech, but it serves as an oracle baseline for evaluating the performance of MARSE with noncausal decoding.

Non-causal decoding with random index selection. We also consider the most naive non-causal decoding policy based on random index selection, as used in MAR models [14]. In this policy, given its cardinality, at each iteration $i \in \{ 1 , . . . , N \}$ } the set $\bar { M } ( i )$ is filled with indices chosen randomly within the set of remaining masked frames $\{ 1 , . . . , T \} \setminus V ( i )$ , following a uniform distribution. This policy can be interpreted as randomly partitioning $\{ 1 , . . . , T \}$ into N blocks of size defined by the cosine schedule function γ, and modeling the probabilistic dependencies between these blocks with (1).

## 2.3. Training

The training of the MARSE model is illustrated in the right part of Fig. 1. Given a labeled dataset of noisy-clean speech pairs $\displaystyle ( \mathbf { y } , \mathbf { x } )$ it is done by optimizing the negative log-likelihood defined from (2) (averaged over the training set). This is equivalent to minimizing the following MSE-based loss function:

$$
\mathcal { L } ( \boldsymbol { \theta } ; \mathbf { x } ) = \mathbb { E } _ { \mathcal { U } ( i ; \{ 1 , \dots , N \} ) } \left[ \sum _ { t \in M ( i ) } \left\| \mathbf { x } _ { t } - f _ { \boldsymbol { \theta } , t } \left( \mathbf { y } , \mathbf { x } _ { V ( i ) } \right) \right\| _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where summation over $i \in \{ 1 , . . . , N \}$ has been equivalently replaced by an expectation. In masked generative modeling, it is common to approximate the expectation using one single sample i [18]. We follow this approach, where the number of masked frames is computed using the same cosine schedule as defined before for the decoding policies, i.e. card $\mathsf { l } ( M ( i ) ) = \lfloor \gamma ( r ) \cdot T \rfloor$ where $r = ( i -$ $1 ) / N$ . In practice, r is sampled from $\mathcal { U } ( [ 0 , 1 [ )$ , which is consistent with $r = ( i - 1 ) / N$ and $i \sim \mathcal { U } ( \{ 1 , { \stackrel { . . . } { \dots } } , { \ddot { N } } \} )$ as $N \to + \infty .$ Given card(M(i)), the set $M ( i )$ is then defined randomly within $\{ 1 , . . . , T \}$ , following a uniform distribution. Finally, $V ( i )$ is set to $\{ 1 , . . . , T \} \setminus M ( i )$

## 2.4. Mitigating exposure bias with quantization

At inference, for each iteration $i \in \{ 1 , . . . , N \}$ of the decoding process, we compute the prediction of the clean speech frames $\mathbf { x } _ { M ( i ) } =$ $\{ f _ { \boldsymbol { \theta } , t } \bigl ( \mathbf { y } , \mathbf { x } _ { V ( i ) } \bigr ) \} _ { t \in M ( i ) }$ , which corresponds to the mean of the Gaussian distributions in (2). In the next iteration, these predicted frames are reinjected into the model as visible frames: $\mathbf { x } _ { V ( i + 1 ) } = \mathbf { x } _ { V ( i ) } \cup$ $\mathbf { x } _ { M ( i ) }$ with $\mathbf { x } _ { V ( 1 ) } = \varnothing$ . This is different from the training process, where the frames $\mathbf { x } _ { V ( i ) }$ in the loss function (3) are obtained from ground-truth clean speech. This mismatch between training and testing conditions, commonly referred to as exposure bias [19], can result in error accumulation during decoding iterations. To mitigate this problem, as proposed in [6], we leverage the fact that we work in the latent space of a NAC to quantize the visible frames $\mathbf { x } _ { V ( i ) }$ using the pretrained NAC quantizer, before feeding them to the model $f _ { \theta } .$ . This is done both at training and inference.

## 3. EXPERIMENTS

This section describes the experiments we conducted to assess MARSE on continuous NAC representations with the different decoding policies, under the same general configuration (in terms of NAC representation, dataset, DNN architecture, training setup, and evaluation metrics).

## 3.1. Experimental setup

NAC representation. Regarding the NAC representation, we used DAC [17], with continuous latent representations of dimension $D =$ 1024. DAC is one of the most widely used NACs, based on a convolutional encoder-decoder architecture and a 12-stage residual vector quantizer (RVQ), whose codebooks each contain 1024 codewords.

Data. For MARSE training and validation, we used the 212-hour train-360 and 11-hour dev subsets of Libri1Mix, respectively. This dataset is obtained by discarding one of the two speakers in Libri2Mix mixtures [22]. Each subset of Libri1Mix contains paired noisy and clean utterances at 16 kHz generated by mixing clean speech from LibriSpeech [23] with noise from WHAM! [24]. We used the 11-hour test subset of Libri1Mix to evaluate in-domain performance. To further evaluate out-of-domain performance, we created a 4-hour synthetic dataset, here referred to as LibriDE-MAND, by mixing clean speech from the train-clean-100 subset of LibriSpeech with noise from DEMAND [25] with a very similar procedure to Libri1Mix. For training, a 1-second clip is randomly cropped from each utterance to form a sequence of embedding vectors with length $T = 5 0$ . At inference, longer utterances are segmented into 1 s-segments that are processed independently by the model.

DNN architecture. We used a Conformer architecture [16] with 16 Conformer blocks, hidden dimension set to 384, 12 attention heads, each of dimension 32, convolution kernel of size 10, expansion factor for convolution module set to 2, and expansion factor for feedforward module set to 4. To align the dimension of DAC embedding vectors with the hidden dimension of the Conformer, two learnable linear layers are applied before and after the Conformer.

Training setup. The model was trained by multi-GPU batch gradient descent, using the AdamW optimizer [26] with a learning rate of $1 0 ^ { - 3 }$ , beta coefficients of (0.9, 0.95), and weight decay of 0.05. The batch size was set to 128 for each GPU and the number of epochs was fixed to 300. The learning rate was linearly increased from 0 to the target value over 10 epochs and followed a cosine decay schedule over the remaining 290 epochs. The multi-GPU training was implemented with the distributed data parallel module of PyTorch across four NVIDIA RTX A100 GPUs with 40GB VRAM.

Evaluation metrics. Since NAC outputs are not aligned samplewise with the clean speech reference due to adversarial training, following prior works [6, 7], the enhanced speech quality was measured using the non-intrusive DNSMOS P.835 SIG, BAK and OVRL scores [27] (between 1.0 and 5.0, the higher the better), which measure respectively the quality of the speech signal, the intrusiveness of background noise, and the overall quality. Speech intelligibility was measured by an ASR-based proxy for phonetic preservation, which is the differential word error rate (dWER in %, the lower the better) between the transcriptions obtained from the enhanced and ground-truth speech. We used a pretrained ASR model based on wav2vec $2 . 0 [ 2 8 ] . ^ { 3 }$ As for computational cost analysis, we measured the amount of giga floating-point operations (GFLOPs), including the encoding and decoding by DAC.

Baselines. As conventional non-AR SE baselines, we used ConvTasNet [20] and DPTNet [21], using the publicly-available models pretrained on Libri1Mix.<sup>4</sup> We also compare MARSE against two NAC-based SE models from [6] that we retrained. Those do not use masking at training or inference, although they have a specific decoding policy: (i) C-NAR is a non-AR model using the ‘all at once’ decoding corresponding to $N = 1$ , and (ii) C-AR is a frame-wise causal AR model, corresponding to $N = T$ Concretely, C-NAR was trained and inferred using a single-step noisy-to-clean mapping, whereas C-AR was trained with a next-frame prediction objective and inferred via frame-wise causal AR decoding of clean representations. Note that we do not use token-based baselines, as it has been shown in [6] that C-NAR and C-AR significantly outperform their counterparts using token representations.

## 3.2. Results

Table 1 provides the results obtained by the proposed MARSE model with the three decoding policies presented in Section 2.2, using $N =$ 10 decoding steps, as well as those obtained by the baselines. Causal decoding is denoted ‘MARSE-causal’. Non-causal decoding with random index selection is denoted ‘MARSE-NC-random’ and constitutes the practical non-causal baseline. Non-causal decoding with oracle index selection is denoted ‘MARSE-NC-oracle’ and is reported as an oracle baseline. Regarding the non-AR baselines, ConvTasNet and DPTNet, the SE quality measured by SIG and OVRL was globally lower than for the other models, but their computational cost was very reasonable. The C-AR and C-NAR baselines showed contrasting performance on the in-domain Libri1Mix. C-AR exhibited higher quality $( \mathrm { S I G } = 3 . 6 4 , \mathrm { B A K } = 4 . 1 1 , \mathrm { O V R L } = 3 . 3 7 )$ but slower inference (GFLOPs = 3856), while C-NAR exhibited lower quality $\mathrm { ( S I G = 3 . 6 0 }$ , BAK = 4.08, OVRL = 3.32) but faster inference $( \mathrm { G F L O P s } = 1 2 3 5 )$ . This observation is consistent with the assumption behind these models: C-AR fully models the causal dependency of speech frames at the expense of computational cost, as it requires T forward passes in the model for inference, while C-NAR independently models each frame for the benefit of computational cost, as it requires a single forward pass in the model for inference. In general, the MARSE model, which assumes inter-block dependency but inner-block independence, demonstrated intermediate SE performance between C-NAR and C-AR, and its computational cost was also in between $( \mathrm { e . g . , S I G = 3 . 6 2 , B A K = 4 . 1 0 , O V R L = 3 . 3 4 } ,$ $\mathrm { G F L O P s = 1 9 1 2 }$ for MARSE-causal). This provides an illustration of the flexibility of the proposed MARSE method in terms of tradeoff between SE performance and computational cost.

<table><tr><td></td><td colspan="5">Libri1Mix</td><td colspan="5">LibriDEMAND</td></tr><tr><td>Model</td><td>SIG ↑</td><td>BAK↑</td><td>OVRL ↑</td><td>dWER↓</td><td>GFLOPs ↓</td><td>SIG ↑</td><td>BAK↑</td><td>OVRL↑</td><td>dWER↓</td><td>GFLOPs↓</td></tr><tr><td>Noisy</td><td>2.46</td><td>1.81</td><td>3.08</td><td>30.19</td><td>1</td><td>2.55</td><td>2.10</td><td>1.92</td><td>21.77</td><td>-</td></tr><tr><td>ConvTasNet [20]</td><td>3.47</td><td>4.10</td><td>3.22</td><td>9.79</td><td>49</td><td>3.41</td><td>3.85</td><td>3.02</td><td>10.49</td><td>51</td></tr><tr><td>DPTNet [21]</td><td>3.44</td><td>4.08</td><td>3.16</td><td>10.05</td><td>12</td><td>3.38</td><td>3.84</td><td>2.98</td><td>9.57</td><td>12</td></tr><tr><td>C-NAR [6]</td><td>3.60</td><td>4.08</td><td>3.32</td><td>12.84</td><td>1235</td><td>3.55</td><td>3.76</td><td>3.08</td><td>9.61</td><td>1334</td></tr><tr><td>C-AR [6]</td><td>3.64</td><td>4.11</td><td>3.37</td><td>20.89</td><td>3856</td><td>3.55</td><td>3.76</td><td>3.09</td><td>15.91</td><td>4166</td></tr><tr><td>MARSE-causal</td><td>3.62</td><td>4.10</td><td>3.34</td><td>12.68</td><td>1912</td><td>3.54</td><td>3.80</td><td>3.11</td><td>9.35</td><td>2065</td></tr><tr><td>MARSE-NC-random</td><td>3.62</td><td>4.09</td><td>3.34</td><td>13.39</td><td>1912</td><td>3.54</td><td>3.83</td><td>3.12</td><td>10.13</td><td>2065</td></tr><tr><td>MARSE-NC-oracle</td><td>3.61</td><td>4.08</td><td>3.34</td><td>12.87</td><td>1912</td><td>3.57</td><td>3.83</td><td>3.14</td><td>8.86</td><td>2065</td></tr></table>

Table 1. Performance obtained by the proposed MARSE model for $N = 1 0$ iterations and the baselines on the Libri1Mix in-domain dataset and LibriDEMAND out-of-domain dataset. The Metrics are described in the text. The scores for input noisy speech (‘Noisy’ item) are provided as a lower reference. Best and second-best scores in each column are bold and underlined.

![](images/5b6a9b523e8252027837664b45573f08963baaad934bda2c71ae5be2b16fe6bc.jpg)  
Fig. 2. DNSMOS OVRL score obtained by the proposed MARSE model (for the three decoding policies) and the C-AR and C-NAR baselines on a subset of 300 samples from Libri1Mix test, as a function of the number of decoding iterations.

Another noticeable result shown in Table 1 is that the MARSE models exhibited intelligibility clearly better than C-AR and even competitive with C-NAR on both the in-domain Libri1Mix (e.g., dWER is 12.68% for MARSE-causal, 20.89% for C-AR, and 12.84% for C-NAR) and out-of-domain LibriDEMAND (e.g., dWER is 9.35% for MARSE-causal, 15.91% for C-AR, and

9.61% for C-NAR). Among the three MARSE models, the practical MARSE-causal performed the best on in-domain data, whereas the non-practical MARSE-NC-oracle exhibited the best scores on out-of-domain data. This is likely because MARSE-causal better models the temporal dependencies in clean speech on in-domain data, while accumulated errors across decoding iterations are more pronounced on out-of-domain data, making a more reliable decoding policy necessary. These results suggest that better non-causal frame-selection strategies could be worth exploring in future work.

Fig. 2 displays the DNSMOS OVRL score obtained by the proposed MARSE model (for the three decoding policies) as a function of the number of decoding iterations $N \in \{ 1 , 5 , 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \}$ To limit computational cost and duration of the experiments, these scores were evaluated on a subset of 300 samples randomly selected from Libri1Mix test, therefore they can be different from those of Table 1. Even though the absolute score difference among decoding policies remains moderate, we can observe in Fig. 2 an improvement in SE quality as the number of decoding steps increases. In particular, it is noticeable that the OVRL score of MARSE-causal closely approaches that of C-NAR for N = 1 and that of C-AR for $N = 5 0$ . This result is consistent with the fact that for N = 1, both MARSE and C-NAR employ ‘all-at-once’ frame decoding, and for $N = 5 0$ , both MARSE-causal and C-AR employ frame-wise causal AR decoding. Between these two ‘extreme’ decoding policies, the MARSE-causal performance improves between C-NAR and C-AR as the number of iterations increases. This confirms the flexibility of the MARSE framework for exploring variants of decoding policies and enabling a trade-off between SE performance and computational cost. In particular, we see in Fig. 2 that the performance of MARSEcausal starts to stagnate in the range 20–40 iterations. Setting N in this range would thus provide a good trade-off between performance and computational cost. Note that all MARSE models were trained with the exact same strategy presented in Section 2.3.

## 4. CONCLUSION

In this work, we proposed the MARSE method and explored different iterative decoding policies for SE using continuous NAC representations. The experimental results showed the ability of this method to provide a flexible trade-off between SE performance and computational cost. Future work may focus on developing a nonoracle confidence measure to determine a more effective non-causal decoding order than random index selection, which may be beneficial in terms of generalization, as suggested by the results obtained with MARSE-NC-oracle on the out-of-domain data.

## 5. REFERENCES

[1] J. Benesty, S. Makino, and J. Chen, Speech enhancement. Springer Science & Business Media, 2006.

[2] P. C. Loizou, Speech enhancement: Theory and practice. CRC press, 2013.

[3] D. Wang and J. Chen, “Supervised speech separation based on deep learning: An overview,” IEEE Trans. Audio, Speech, Lang. Proc., vol. 26, no. 10, pp. 1702–1726, 2018.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez et al., “Attention is all you need,” Adv. Neural Inform. Proc. Syst. (NeurIPS), vol. 30, 2017.

[5] P. Mousavi, G. Maimon, A. Moumen, D. Petermann, J. Shi, H. Wu et al., “Discrete audio tokens: More than a survey!” Trans. Mach. Learn. Res. (TMLR), Jun. 2025.

[6] S. Kammoun, X. Alameda-Pineda, and S. Leglaive, “Modeling strategies for speech enhancement in the latent space of a neural audio codec,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2026.

[7] Z. Wang, X. Zhu, Z. Zhang, Y. Lv, N. Jiang, G. Zhao, and L. Xie, “SELM: Speech enhancement using discrete tokens and language models,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2024.

[8] J. Yao, H. Liu, C. Chen, Y. Hu, E. Chng, and L. Xie, “GenSE: Generative speech enhancement via language models using hierarchical modeling,” in Int. Conf. Learn. Rep. (ICLR), 2024.

[9] H. Xue, X. Peng, and Y. Lu, “Low-latency speech enhancement via speech token generation,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2024.

[10] H. Yang, J. Su, M. Kim, and Z. Jin, “Genhancer: High-fidelity speech enhancement via generative modeling on discrete codec tokens,” in Interspeech, 2024.

[11] X. Li, Q. Wang, and X. Liu, “MaskSR: Masked language model for full-band speech restoration,” in Interspeech, 2024.

[12] J. Zhang, J. Yang, Z. Fang, Y. Wang, Z. Zhang, Z. Wang et al., “AnyEnhance: A unified generative model with promptguidance and self-critic for voice enhancement,” IEEE Trans. Audio, Speech, Lang. Proc., vol. 33, pp. 3085–3098, 2025.

[13] T. H. Pham, T. D. Nguyen, P. T. Tran, J. S. Chung, and D. D. Nguyen, “MAGE: A coarse-to-fine speech enhancer with masked generative model,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2026.

[14] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive image generation without vector quantization,” in Adv. Neural Inform. Proc. Syst. (NeurIPS), 2024.

[15] H. Yang, G. Wichern, R. Aihara, Y. Masuyama, S. Khurana, F. G. Germain, and J. Le Roux, “Investigating continuous autoregressive generative speech enhancement,” in Interspeech, 2025, pp. 2360–2364.

[16] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented Transformer for speech recognition,” in Interspeech, 2020.

[17] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved RVQGAN,” Adv. Neural Inform. Proc. Syst. (NeurIPS), vol. 36, 2023.

[18] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman, “MaskGIT: Masked generative image transformer,” in IEEE/CVF Conf. Computer Vision Pattern Recog. (CVPR), 2022.

[19] M. Ranzato, S. Chopra, M. Auli, and W. Zaremba, “Sequence level training with recurrent neural networks,” in Int. Conf. Learn. Rep. (ICLR), 2016.

[20] Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation,” IEEE Trans. Audio, Speech, Lang. Proc., vol. 27, no. 8, pp. 1256–1266, 2019.

[21] J. Chen, Q. Mao, and D. Liu, “Dual-path Transformer network: Direct context-aware modeling for end-to-end monaural speech separation,” in Interspeech, 2020.

[22] J. Cosentino, M. Pariente, S. Cornell, A. Deleforge, and E. Vincent, “LibriMix: An open-source dataset for generalizable speech separation,” preprint arXiv:2005.11262, 2020.

[23] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR corpus based on public domain audio books,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2015.

[24] G. Wichern, J. Antognini, M. Flynn, L. R. Zhu, E. McQuinn, D. Crow et al., “WHAM!: Extending speech separation to noisy environments,” in Interspeech, 2019.

[25] J. Thiemann, N. Ito, and E. Vincent, “The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings,” Proc. Meetings on Acoustics, vol. 19, no. 1, p. 035081, 2013.

[26] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Int. Conf. Learn. Rep. (ICLR), 2018.

[27] C. Reddy, V. Gopal, and R. Cutler, “DNSMOS P.835: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors,” in IEEE Int. Conf. Acoust., Speech, Sig. Proc. (ICASSP), 2022.

[28] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Adv. Neural Inform. Proc. Syst. (NeurIPS), vol. 33, 2020.