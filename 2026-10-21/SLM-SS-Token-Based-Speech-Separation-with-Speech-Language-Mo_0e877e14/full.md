# SLM-SS: Token-Based Speech Separation with Speech Language Models

Chenda Li<sup>∗,†</sup>, Member, IEEE, Tianhua Li<sup>∗</sup>, Yiming Chen, Wei Wang, Jiahe Wang, Xin Zhou, Yanmin Qian<sup>†</sup>, Senior Member, IEEE

Abstract—Speech separation (SS) has witnessed substantial progress in recent years with the advent of neural networkbased methods, achieving notable improvements on signal-level evaluation metrics. Existing speech separation systems are mostly optimized with signal-level objectives, and often yield outputs that are acceptable in waveform metrics but suboptimal for intelligibility-sensitive downstream tasks. To address this gap, we propose SLM-SS, a token-based generative speech separation framework that models multi-talker mixtures with speech language models. We first discretize clean speech into multicodebook token sequences with neural codecs and serialize multi-speaker targets via Serialized Output Training (SOT). An encoder-decoder autoregressive speech language model predicts the zero-order sequence to establish speaker ordering and coarse linguistic content, and a non-autoregressive module then refines higher-order residual codebooks for efficient reconstruction. We further introduce a per-talker non-autoregressive strategy that simplifies high-order modeling and improves the stability. On Libri2Mix, SLM-SS with per-talker NAR achieves 6.92% WER, approaching codec-only reconstruction. The WavTokenizer variant eliminates NAR refinement and achieves the highest mean subjective quality score among evaluated separation systems. Its UTMOS and DNSMOS advantages over discriminative baselines persist on unseen noisy and Chinese mixtures without fine-tuning.

Index Terms—speech language model, speech separation, speech intelligibility.

## I. INTRODUCTION

PEECH separation (SS) is a fundamental problem in S multi-talker speech processing, aiming to isolate individual speech sources from overlapping-signal mixtures [1]. It plays a critical role in a wide range of real-world applications, including multi-talker automatic speech recognition (ASR), speaker identification (SID), and assistive technologies such as hearing aids. Currently, SS has been predominantly addressed using discriminative approaches [1]–[6] , which are commonly optimized with signal-level objectives such as scaleinvariant signal-to-distortion ratio (SI-SDR) [7]. Although these methods have demonstrated strong performance in terms of waveform reconstruction and objective evaluation metrics, the separated signals often contain distortions or artifacts, which may have some impact on speech intelligibility and linguistic structure, substantially degrading the performance of downstream tasks such as ASR and SID [8]. In contrast, generative methods [9]–[12], explicitly model the underlying data distribution of clean speech signals, thereby offering greater potential to produce separated outputs that are more coherent and intelligible. Despite these advantages, existing generative approaches face notable challenges in practice. In particular, the diffusion- or flow-based separation methods [10], [11] typically rely on iterative sampling procedures, leading to slow decoding speeds that limit scalability and real-time applicability. Moreover, such models are prone to hallucination [13], generating non-existent or spurious speech content, further limiting their reliability in downstream applications.

Meanwhile, recent advances in speech language models (SLMs) [14] have demonstrated strong potential to bridge acoustic and linguistic representations through discrete speech tokens. The SLMs utilize neural codecs [15]–[18] to convert continuous speech signals into quantized token sequences. The neural codecs employ residual vector quantization (RVQ) [19], [20] to compress continuous-space audio into discrete codebook sequences at extremely low bit rates. Thus, these SLMs enable speech to be processed in an autoregressive next-token prediction manner, allowing powerful sequence modeling techniques originally developed for text language models to be directly applied to speech. As a result, SLM-based approaches have achieved notable progress across a range of intelligibilityoriented speech processing tasks, including ASR [21], textto-speech (TTS) [22], [23], speech enhancement (SE) [24], conditional target source extraction [25], and unified multitask modeling. In these settings, discrete token representations provide a natural interface for integrating acoustic structure with higher-level semantic information, leading to improved modeling consistency and robustness. Collectively, these developments suggest that discrete SLM-based modeling offers a promising alternative to conventional signal-level optimization, and motivate its extension to more challenging problems such as speech separation.

Building on our ICASSP 2026 work, SLM-SS [26], we present an extended SLM-SS framework that leverages speech language models (SLMs) together with quantized neural codecs to improve the intelligibility of separated speech. We adopt Encodec [16] to convert continuous waveforms into discrete multi-codebook token sequences, which makes speech separation amenable to language-modeling style sequence prediction. To represent multiple sources, we concatenate per-speaker token streams using Serialized Output

Training (SOT) [27], following recent SOT-based ASR developments [28], [29]. The resulting sequence is generated with a transformer autoregressive (AR) encoder-decoder: the encoder extracts mixture representations, and the decoder aligns mixture features with target tokens via cross-attention. To further improve quality and reduce latency, we augment the AR backbone with a non-autoregressive (NAR) module that predicts higher-order codebooks conditioned on lower-order ones using independent, one-to-one token embeddings; we also propose a more efficient NAR variant to reduce training complexity. Compared with the ICASSP version, this paper provides a more complete formulation of hybrid AR/NAR modeling, introduces a per-talker NAR variant for efficient high-order token prediction, supports variable numbers of talkers, and includes more comprehensive codec comparisons as well as substantially expanded objective and subjective evaluations. Overall, the proposed framework offers a generative alternative to conventional SS approaches, improving speech intelligibility and yielding strong performance on downstream tasks.

Our contributions can be summarized as follows:

• We introduce SLM-SS, a generative speech separation framework that performs intelligibility-aware separation by predicting discrete neural-codec tokens with a speech language model.

• We develop a hybrid AR/NAR generation scheme for multi-codebook token prediction, including a per-talker NAR variant that improves high-order codebook generation while keeping decoding efficient.

• We compare two representative neural codecs (Encodec [16] and WavTokenizer [15]) and highlight the benefits of a single-codebook codec design, which simplifies training by removing the need for an additional non-autoregressive (NAR) decoding module.

• We extend SLM-SS to support a variable number of talkers, improving flexibility in practical multi-speaker scenarios.

• We conduct extensive objective and subjective evaluations, revealing complementary strengths of generative and discriminative separation approaches across linguistic, signal-level, and perceptual metrics.

## II. RELATED WORKS

## A. Speech Separation

Speech separation aims to isolate clean, single-talker speech from overlapping multi-talker mixtures that may also contain background noise. Most discriminative methods formulate separation as learning masking [1] or mapping [30] functions under a fixed and known number of speakers (Fig. 1(a)). By estimating and applying different masks to intermediate audio representations, these approaches achieve the separation and time-domain separation.

TF-domain separation performs separation on a spectrogram (e.g., STFT). It predicts masks or mapped spectra over time–frequency bins and then reconstructs waveforms with an inverse transform, where phase handling is often an additional concern. Representative examples include Deep-CASA [3], which performs frame-level spectral separation with permutation-invariant training (PIT) [31], [32] and uses Deep Clustering [2] for talker tracking, and BSRNN [5], which explicitly decomposes a mixture spectrogram into subbands and interleaves band-level and sequence-level modeling for improved performance.

In contrast, time-domain separation directly operates on waveforms or learned time-domain latent representations without an explicit TF transform, thereby avoiding explicit phase reconstruction. TasNet [33] adopts an encoder–separator– decoder framework and performs separation on nonnegative latent representations, while Conv-TasNet [4] introduces a convolutional encoder to learn waveform representations optimized for separation and recovers sources via mask-based weighting. Subsequent works [6], [34]–[36] further improve time-domain modeling by capturing both short- and long-term dependencies with a multi-scale transformer-based architecture.

Beyond fixed-speaker settings, TUSS [37] and Flexio [38] explore multi-domain separation with a variable number of learnable prompts to specify target sources. SepEDA [39], SD-SepEDA [40] and MUSE [41] use triple-path processing blocks and attractor generation to support a variable number of talkers, while EEND-EDA [42] integrates the Encoder-Decoder Attractor (EDA) module into an end-to-end speaker diarization (SD) system to handle variable-talker scenarios. More broadly, unified source separation and extraction has been explored in USE [43] and Promptsep [44], indicating a trend toward general and scalable separation frameworks that can dynamically adjust their behavior across multiple separation tasks.

![](images/68caeba26fb3a55b150e1c505cb83bd91fd621f001c4df3aaadbbd89963aaaf6.jpg)  
(b) Speech language model method

![](images/b301e3e8604f910cdd7b6963d7ddd37f43a2135a6ca6770f7162d8edd3245a63.jpg)  
Fig. 1. Structure of discriminative model and SLM model for speech separation.

## B. Language Model in Speech Processing

Recent studies have demonstrated that combining SLMs with discrete feature representations is effective across many speech processing tasks. For text-to-speech synthesis (TTS) task, VALL-E [22] and Seed-TTS [45] merge discrete audio tokens from neural codecs with text tokens to build a unified vocabulary. They then use an LM to generate stylistically consistent speech via autoregressive continuation conditioned on text and audio prompts. For speech recognition, Seed-ASR [46] and FireRedASR [47] incorporate LLMs to achieve state-ofthe-art ASR performance. For target speaker extraction (TSE), TSELM uses discrete WavLM outputs [48] to encode targetspeaker information and reconstruct speech from tokens [25]. UniSep models SoundStream-derived discrete sequences [49] and uses prompt audio for conditional target sound separation [50]. For general audio separation, OpenSep [51] leverages descriptive captions produced by an external audio captioning model to capture source-specific characteristics. SepALM targets noise-related issues by using an LM for endto-end denoising of separated speech [52]. SELM [53] extracts WavLM-based discrete representations and models them with a speech language model for speech enhancement (SE). Highfidelity token-based speech enhancement further extends this line of work [24]. TokenSplit introduces a multitask masking framework that jointly performs speech separation, ASR, and speech synthesis, optionally conditioned on transcripts [54].

Despite these advances, most SLM-based approaches focus on single-stream modeling or conditional extraction, rather than structured multi-talker separation with explicit sequencelevel constraints. As illustrated in Fig. 1(b), we instead model multi-talker speech separation end-to-end, without conditional prompts or external priors. Our preliminary ICASSP study (SLM-SS [26]) validated the core idea of casting separation as discrete multi-codebook sequence generation. This journal version substantially extends SLM-SS in both modeling and evaluation:

• Improved decoding/training for better quality– efficiency trade-off. Beyond the VALL-E-inspired training recipe used in ICASSP, we explicitly consider speech quality and training cost, and propose a per-talker nonautoregressive (NAR) variant to enable efficient highorder decoding.

• Codec comparison and simplified training. While the ICASSP work used only Encodec [16], this journal version systematically compares different discrete tokenizers/codecs and analyzes their impact on separation performance. In particular, by adopting WavTokenizer [15], which produces a single-codebook token sequence, we no longer require the NAR module, substantially simplifying the overall training pipeline.

• Variable-talker experiments. We evaluate scalability and generalization to mixtures with varying numbers of speakers.

• More detailed ablations and comprehensive evaluation. We conduct finer-grained ablations to quantify the effects of different autoregressive decoding strategies and other key design choices. We also add both objective metrics and subjective listening tests, and report results across multiple codecs.

## III. SPEECH LANGUAGE MODEL FOR SPEECH SEPARATION A. Overview

Fig. 2 summarizes the overall pipeline of SLM-SS. As shown in Fig. 2(a), we first construct discrete supervision by quantizing each reference single-talker waveform with Encodec [16] into a multi-codebook token representation. Encodec employs residual vector quantization: the zero-order codebook carries the most salient spectral-temporal structure, while higher-order codebooks encode the remaining resid ual details. We then serialize tokens from different talkers into a single target sequence using serialized output training (SOT) [27]. As shown in Fig. 2(b), we learn an attentionbased encoder-decoder (AED) model to predict the zeroorder SOT sequence in an autoregressive (AR) manner, using cross-attention over mixture features to align the mixture with the target tokens. Then, a non-autoregressive (NAR) model (Fig. 2(c)) with the same backbone is further used to predict higher-order codebooks conditioned on all lower-order sequences, serving as a refinement stage. This AR-NAR decomposition follows the spirit of VALL-E [22]: AR modeling focuses on the most informative codebook, while NAR modeling completes the higher-order residual codebooks. Finally, as shown in Fig. 2(d), we split the serialized sequence into per-talker streams using the special speaker change (<SC>) tokens, and decode the resulting multi-codebook tokens with the Encodec decoder to obtain the separated single-talker waveforms.

## B. Training Label Preparation

We formulate speech separation as discrete sequence generation. We use Encodec [16] to quantize each single-talker waveform into a M-order $( M = 3 2 )$ in Encodec) codebook representation, where each m-th $( m = 0 , 1 , \ldots , M - 1 )$ ) order codebook has a base vocabulary $\nu _ { m } ^ { ' }$ with size $| \nu _ { m } ^ { \prime } | = 1 0 2 4$ For serialized output training (SOT), we further augment each codebook vocabulary with three special symbols (<SOS>, <SC>, <EOS>), and denote the resulting vocabulary by:

$$
\mathcal { V } _ { m } = \mathcal { V } _ { m } ^ { ' } \cup \{ < S 0 S > , < S C > , < E 0 S > \} .\tag{1}
$$

Note that different-order codebooks all have the same base vocabulary size $| \mathcal { V } _ { m } | ~ = ~ 1 0 2 7$ (hence the same token-ID range); however, token IDs are defined independently for each order $m ,$ so the same numerical ID across orders is not comparable and carries no shared meaning. In contrast, the special symbols maintain consistent semantic meanings across all M codebooks, enabling unambiguous representation of structural information such as talker transitions and sequence boundaries.

Our training set is constructed via data simulation: each multi-talker mixture is synthetically generated by mixing multiple clean single-talker utterances, so the underlying clean sources are known and can be directly used as supervision. As Fig. 2(a) shows, for each simulated mixture, we independently encode every constituent clean utterance with Encodec to obtain its multi-codebook token sequences. To prepare a single label sequence for multi-talker training, we adopt serialized output training (SOT) [27] to serialize talker-wise token streams into one target O:

$$
\mathbf { O } = [ \mathbf { o } _ { 0 } , \mathbf { o } _ { 1 } , \dots , \mathbf { o } _ { M - 1 } ] ,\tag{2}
$$

![](images/7a592cd968b6ab0e8302a9b83599964e4680d9f52710e75537d5ff2225ce31a3.jpg)  
Fig. 2. Overview of SLM-SS. (a) Encodec quantizes single-talker audio into multi-codebook sequences, then merges them using SOT for the training label preparation. (b) The AR model predicts the zero-order sequence. (c) NAR model sequentially predicts higher-order codebook sequences given lower-order ones. (d) Segment the SOT sequences into single-person sequences, then decoded into audio, respectively.

where $\mathbf { o } _ { m }$ is the m-th order codebook training label. For the m-th order codebook sequence $\mathbf { o } _ { m }$ , we prepend <SOS>, concatenate the token sequences of different talkers in utterance first-in-first-out order, insert <SC> between adjacent talkers, and append <EOS>:

$$
\begin{array} { r l } & { \mathbf { o } _ { m } = \left[ < \mathrm { S O S } > , r _ { 1 , m } ^ { 1 } , \ldots , r _ { K ^ { 1 } , m } ^ { 1 } , < \mathrm { S C } > , \right. } \\ & { \quad \quad \left. r _ { 1 , m } ^ { 2 } , \ldots , r _ { K ^ { 2 } , m } ^ { 2 } , < \mathrm { S C } > , \ldots , < \mathrm { S C } > , \right. } \\ & { \quad \quad \left. r _ { 1 , m } ^ { J } , \ldots , r _ { K ^ { J } , m } ^ { J } , < \mathrm { E O S } > \right] . } \end{array}\tag{3}
$$

where $r _ { k , m } ^ { j } \in \mathcal { V } _ { m } ^ { ' }$ denotes the k-th token of the $j \mathrm { - t h }$ talker in the m-th order codebook sequence, and $K ^ { j }$ represents the token sequence length corresponding to the j-th talker.

## C. AR Model for Zero-order Codebook Sequence Decoding

During training, the AED model is required to predict the zero-order codebook sequence, which carries the dominant spectral and temporal structure of the target speech. For the encoder, we adopt a pre-trained WavLM [48] model and finetune it for speech separation. For the decoder, we follow the architectural design of the Whisper [55] decoder and train it from scratch for discrete token generation.

Given a multi-talker input waveform, the encoder extracts hidden representations from all WavLM layers. We then use a linear fusion layer followed by layer normalization to aggregate them into a unified acoustic representation $\mathbf { H } _ { \operatorname* { m i x } } ^ { \mathrm { A R } }$ . The AR model mainly learns the zero-order codebook at $m = 0$ . For the sake of convenience in description, we rewrite the training label $\mathbf { o } _ { 0 }$ in Eq.3 into the following form when $m = 0 \colon$

$$
\mathbf { o } _ { 0 } = [ o _ { 0 } ^ { 1 } , \dots , o _ { 0 } ^ { N } ] ,\tag{4}
$$

where $o _ { 0 } ^ { n }$ is the n-th token in the training label $\mathbf { o } _ { 0 }$ and $N$ is the length of $\mathbf { o } _ { 0 }$ . During training, conditioned on $\mathbf { H } _ { \operatorname* { m i x } } ^ { \mathrm { A R } }$ and the teacher-forced history $[ o _ { 0 } ^ { 1 } , \ldots , o _ { 0 } ^ { n - 1 } ]$ , the autoregressive decoder is trained to predict the next token $\hat { o } _ { 0 } ^ { n }$

$$
\mathbf { q } _ { 0 } ^ { n } = \mathrm { D e c o d e r } ( [ o _ { 0 } ^ { 1 } , \dots , o _ { 0 } ^ { n - 1 } ] , \mathbf { H } _ { \mathrm { m i x } } ^ { \mathrm { A R } } ) .\tag{5}
$$

where $\mathbf { q } _ { 0 } ^ { n } \in \mathbb { R } ^ { | \mathcal { V } _ { 0 } | }$ is the probability distribution of the $n \mathrm { - }$ th token. During inference, we decode a predicted token $\hat { o } _ { 0 } ^ { n }$ from $\mathbf { q } _ { 0 } ^ { n }$ , and the decoder conditions on its own previously generated history $[ \hat { o } _ { 0 } ^ { 1 } , \dots , \hat { o } _ { 0 } ^ { n - 1 } ]$ . In this way, the AED model learns to map a multi-talker mixture directly to the zero-order SOT sequence, thereby capturing both talker serialization and coarse speech content in a single autoregressive generation process. We optimize the AR decoder with the standard crossentropy loss:

$$
\mathcal { L } _ { \mathrm { A R } } = - \sum _ { n = 1 } ^ { N } \log q _ { 0 } ^ { n } ( o _ { 0 } ^ { n } ) ,\tag{6}
$$

where one-hot labels are used, and $q _ { 0 } ^ { n } ( o _ { 0 } ^ { n } )$ denotes the predicted probability assigned to the ground-truth token $o _ { 0 } ^ { n }$ (i.e., the $o _ { 0 } ^ { n } .$ -th entry of $\mathbf { q } _ { 0 } ^ { n } )$ .

## D. NAR Model for High-order Codebook Sequences Decoding

Inspired by the AR+NAR decomposition in VALL-E [22], we adopt the same backbone as the AED model and remove the unidirectional attention mask to implement a nonautoregressive (NAR) decoder for high-order codebook prediction. Unlike VALL-E, whose NAR stage is purely conditioned on discrete codec tokens, our NAR model is equipped with an encoder and takes continuous mixture features as input, enabling refined prediction with richer acoustic conditioning. As illustrated in Fig. 3, to enable collaborative training across multiple codebook orders, we design several independent token embedding layers with one-to-one correspondence to different orders of codebook sequences, and share a common positional embedding, allowing different orders to be modeled in a coordinated manner over the entire SOT sequence. In addition, we introduce a task embedding $\mathbf { T } _ { m }$ that is obtained by encoding the target codebook order index $m$ (with m ∈ $\{ 1 , \ldots , M - 1 \} )$ ), explicitly specifying that the current task is to predict the m-th order codebook; this enables the model to effectively distinguish among different prediction tasks during both training and inference.

![](images/79c5ee8e7a2e8cc53d33c19bbbc990ad3d0923be98856a397fe597b7963153bd.jpg)  
Fig. 3. Detail of NAR Decoder. NAR decoder integrates all low-order sequence information by employing multiple independent token embeddings.

When predicting the m-th order codebook sequence, information from all lower-order codebook sequences must be jointly considered. To this end, all lower-order sequences are first embedded independently and then summed together. The positional embedding and the corresponding task embedding are subsequently incorporated to form the total embedding $\mathbf { E } _ { m } .$ , which serves as the input to the NAR model, namely:

$$
\mathbf { E } _ { m } = \left( \sum _ { \ell = 0 } ^ { m - 1 } \operatorname { E m b } \bigl ( \mathbf { o } _ { \ell } ; \mathbf { W } _ { \ell } \bigr ) \right) + \mathbf { P } + \mathbf { T } _ { m } ,\tag{7}
$$

where Emb $\big ( \mathbf { o } _ { \ell } ; \mathbf { W } _ { \ell } \big ) , \ell \in [ 0 , m )$ , denotes the token embedding of the ℓ-th order codebook sequence $\mathbf { o } _ { \ell }$ with the $\ell -$ th token embedding layer $\mathbf { W } _ { \ell } , \mathbf { P }$ represents the positional embedding, and $\mathbf { T } _ { m }$ is a learned task embedding obtained from an encoding of the order index $m \in \{ 1 , \ldots , M - 1 \}$ indicating that the decoder should predict the m-th order codebook. The resulting embedding $\mathbf { E } _ { m }$ is then fed into a stack of Transformer layers for deep contextual modeling. Finally, the output representations $\mathbf { H } _ { \mathrm { o u t } } ^ { m }$ are projected onto the token embedding space of the m-th order codebook $\mathbf { W } _ { m } ,$ yielding the probability distribution $\mathbf { q } _ { m }$ over tokens in the m-th order vocabulary $\nu _ { m }$

$$
\begin{array} { r l } & { \mathbf { H } _ { \mathrm { o u t } } ^ { m } = \mathrm { D e c o d e r T r a n s f o r m e r } ( \mathbf { E } _ { m } ) , } \\ & { \quad \mathbf { q } _ { m } = \mathrm { S o f t m a x } \big ( \mathbf { H } _ { \mathrm { o u t } } ^ { m } \mathbf { W } _ { m } ^ { \top } \big ) . } \end{array}\tag{8}
$$

To train the NAR model, we optimize it with the cross-entropy loss over all high-order codebook sequences. Specifically, for each $m \in \{ 1 , \ldots , M - 1 \}$ , we denote the training label as $\mathbf { o } _ { m } = [ o _ { m } ^ { 1 } , \hdots , o _ { m } ^ { N } ]$ , and let $\mathbf { q } _ { m } ^ { n } \in \mathbb { R } ^ { | \nu _ { m } | }$ be the predicted distribution at position n (i.e., the n-th row of ${ \bf q } _ { m } )$ . The NAR loss can be defined as:

$$
\mathcal { L } _ { \mathrm { N A R } } = - \sum _ { m = 1 } ^ { M - 1 } \sum _ { n = 1 } ^ { N } \log q _ { m } ^ { n } ( o _ { m } ^ { n } ) .\tag{9}
$$

It is noted that outer summation over m in Eq. 9 represents our desired objective of optimizing all high-order codebooks jointly. However, directly computing the full sum over m ∈ $\{ 1 , \ldots , M - 1 \}$ for every training sample is memory-intensive, since each m-th prediction conditions on all lower-order sequences. Therefore, in practice we randomly sample a specific m $\sim \mathrm { U n i f o r m } ( \{ 1 , \dots , M - 1 \} )$ for each training sample and optimize only the corresponding term, which serves as a stochastic approximation of Eq. 9. Meanwhile, because the NAR model must integrate information from all lowerorder codebook sequences to predict higher-order ones, its computational cost and training challenges (e.g., convergence time and sensitivity to hyperparameters) increase with the total order of the codebooks. In this work, we predict codebooks up to the 8-th order out of 32 codebooks in Encodec (i.e., $M = 8 ) ;$ as we will show in Sec. IV, this truncation has a limited impact on performance.

## E. Inference Procedure and Speech Reconstruction

During inference, the trained AED model first receives the multi-talker mixture and autoregressively generates the predicted zero-order codebook sequence $\hat { \bf { o } } _ { 0 }$ in SOT format. Here, we use $\hat { \mathbf { o } } _ { m }$ to denote the decoded (predicted) m-th order codebook sequence, distinguishing it from the groundtruth training label $\mathbf { o } _ { m }$ . The predicted zero-order sequence already determines the talker ordering and the segmentation boundaries through the predicted <SC> symbols. The generated $\hat { \bf { o } } _ { 0 }$ is then used in two ways. First, it provides the loworder input required by the NAR model to predict higherorder codebooks, as described in the next subsection. Second, after all $m$ codebook orders have been obtained, the complete SOT sequence is segmented according to the <SC> symbols to recover the single-person token sequences.

Each segmented single-person predicted multi-codebook sequence is finally passed to the Encodec decoder to reconstruct the separated waveform. Because the number of talkers is implicitly represented by the number of predicted separator symbols <SC>, the proposed framework can naturally handle mixtures with an unknown talker count at inference time, without requiring an additional talker-count estimator or prior assumptions on the number of talkers.

## F. Per-Talker NAR: Simplified NAR Decoding Strategy

As discussed earlier, training joint multi-talker NAR models poses significant challenges due to quadratic growth in computational cost and convergence difficulty as the number of codebooks increases. However, we observe that the most critical information for speech separation is already determined by the zero-order codebook sequence produced by the AR model. Since the zero-order codebook contains dominant spectral and temporal structure, requiring the NAR model to jointly refine codebooks across all speakers may introduce unnecessary complexity.

Motivated by this observation, we propose an alternative per-talker NAR approach. Instead of jointly refining all speak-$\mathrm { e r s ^ { \prime } }$ codebooks in the SOT format, we first segment the multitalker zero-order sequence $\hat { \bf { o } } _ { 0 }$ using the predicted separator symbols <SC>, then independently apply the NAR model to each single-talker segment. This approach leverages the mixture representation together with each talker’s accumulated low-order tokens to predict the corresponding higher-order codebooks. By design, the per-talker NAR avoids modeling speaker transitions at higher orders, substantially reducing task complexity and computational burden. The full inference procedure and notation are summarized in Algorithm 1.

Algorithm 1 Inference with Per-talker NAR Scheme   
Input: Mixture waveform x, number of codebook orders ${ \overline { { M } } } .$   
Output: Reconstructed waveform $\{ \hat { \mathbf { y } } ^ { j } \} _ { j = 1 } ^ { J }$ of each talker .   
Notation:   
$\hat { \mathbf { o } } _ { m } \colon$ predicted m-th order codebook sequence.   
$J _ { \vdots }$ talker count inferred from oˆ<sub>0</sub> by <SC>.   
$\hat { \mathbf { o } } _ { 0 } ^ { j } \colon j \mathbf { - } \mathrm { t h }$ segment of $\hat { \bf { o } } _ { 0 }$ split by <SC>.   
$\mathbf { H } _ { \operatorname* { m i x } } ^ { \mathrm { A R } } / \mathbf { H } _ { \operatorname* { m i x } } ^ { \mathrm { N A R } }$ : mixture features from AR/NAR encoders.   
$\mathbf { Z } ^ { j } \colon$ accumulated tokens for talker j (init $\begin{array} { r } { \hat { \mathbf { o } } _ { 0 } ^ { j } \dag , } \end{array}$ append $\hat { \mathbf { o } } _ { m } ^ { j }$ for m $\geq 1 )$   
1: INFERENCE FUNCTION(x, M)   
2: $\mathbf { H } _ { \mathrm { m i x } } ^ { \mathrm { A R } }  \mathrm { E n c o d e r } _ { \mathrm { A R } } ( \mathbf { x } )$   
3: $\hat { \mathbf { o } } _ { 0 } \gets \mathrm { D e c o d e r } _ { \mathrm { A R } } ( \mathbf { H } _ { \mathrm { m i x } } ^ { \mathrm { A R } } )$   
4: $\{ \hat { \mathbf { o } } _ { 0 } ^ { j } \} _ { j = 1 } ^ { J }  \mathrm { S p l i t } ( \hat { \mathbf { o } } _ { 0 } , < \mathrm { S C } > )$   
5:   
6: ${ \bf H } _ { \mathrm { m i x } } ^ { \mathrm { N A R } }  \mathrm { E n c o d e r } _ { \mathrm { N A R } } ( { \bf x } )$   
7:   
8: for $j = 1 , \dotsc , J$   
9: $\mathbf { Z } ^ { j }  \hat { \mathbf { o } } _ { 0 } ^ { j }$   
10:   
11: for $m = 1 , \ldots , M - 1$   
12: $\hat { \mathbf { o } } _ { m } ^ { j } \gets \mathrm { D e c o d e r } _ { \mathrm { N A R } } ( \mathbf { H } _ { \mathrm { m i x } } ^ { \mathrm { N A R } } , \mathbf { Z } ^ { j } )$   
13: $\mathbf { Z } ^ { j } \gets \mathrm { A p p e n d } ( \mathbf { Z } ^ { j } , \hat { \mathbf { o } } _ { m } ^ { j } )$   
14:   
15: $\hat { \mathbf { y } } ^ { j } \gets \mathrm { D e c o d e r } _ { \mathrm { C o d e c } } ( \mathbf { Z } ^ { j } )$   
16:   
17: return $\{ \hat { \mathbf { y } } ^ { j } \} _ { j = 1 } ^ { J }$

## G. Fully AR Scheme with a Single-Order Codec

Although Encodec provides strong reconstruction quality, its multi-order residual vector quantization design requires the additional NAR stage to recover higher-order codebooks, which increases both training complexity and inference cost. To examine whether the separation framework can be further simplified, we additionally consider a fully autoregressive variant built on a single-order neural audio codec.

Specifically, we replace Encodec with WavTokenizer [15], which represents each utterance using only one codebook sequence while maintaining high-fidelity reconstruction. Under this setting, the AED model directly predicts the complete target token sequence, and no higher-order codebook refinement is required. As a result, the overall system no longer needs the NAR module, and the separation pipeline is reduced to three steps: single-order token generation, SOTbased sequence segmentation, and codec decoding.

This variant serves two purposes. First, it provides a computationally simpler alternative to the Encodec-based AR-NAR framework. Second, it allows us to analyze how codec design, especially the choice between multi-order and singleorder tokenization, affects downstream speech separation performance. Experimental comparisons between this scheme and the Encodec-based methods are reported in Sec. IV.

## IV. EXPERIMENT

## A. Experiment Setup

Model: The encoder is initialized with pre-trained WavLMlarge weights [48], while the decoder adopts the dimensional configuration of Whisper-medium [55], with an adjusted vocabulary size and a reduced depth of 16 Transformer layers to save computational resources. The resulting model comprises approximately 600M parameters in total. Training is performed for 30 epochs using an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , together with cosine annealing decay and linear warm-up during the first three epochs.

For AR decoding, we employ beam search with beamsize = 20, and apply blank suppression as well as Ngram blocking to effectively prevent empty predictions and infinite repetition during inference. For the NAR model and its variant, due to training difficulties arising from multiorder dependency modeling, we adopt a staged training strategy. Specifically, the model is first trained on the first-order codebook prediction task, where the zero-order codebook sequence generated by the AR model is used to predict the first-order codebook sequence. During this stage, embedding layers corresponding to higher-order codebook sequences are excluded from computation. Subsequently, the resulting model weights are retrained on all non-zero codebook prediction tasks using identical training configurations, enabling stable convergence across multiple codebook orders.

Dataset & Baseline: Experiments are conducted on LibriMix [56], max and clean version with 16kHz sampling rate, using the 100h and 360h subsets for training and the test subset for evaluation. We follow the original LibriMix data simulation pipeline for mixture generation, and additionally apply preprocessing to clean speech before Encodec quantization: VAD-based segmentation is used to remove silent regions. This extra step has two purposes. First, it reduces unnecessary silent tokens in codebook sequences and thus lowers computational overhead. Second, and more importantly for SOT modeling, it avoids variable-length silence spans that can shift token positions and even invert the effective temporal order of speakers in the generated SOT sequence. Since Encodec-based encoding and decoding, as well as partial codebook truncation, may introduce distortions that affect signal quality, we explicitly examine the relationship between the original audio and its Encodec-processed counterparts.

Although the clean waveform is the nominal reference, directly modeling all 32 Encodec codebooks is impractical in our AR–NAR setup: when predicting higher-order codebooks, the NAR model must condition on all lower-order codebooks, which substantially increases optimization difficulty and computational cost as the order grows. Following the coarse-to-fine codec language modeling paradigm used in VALL-E [22], we focus on the first 8 codebooks, which retain most intelligibility-relevant and linguistic information, while treating higher-order codebooks mainly as residual detail refinement. Therefore, in our setting, the practical upper bound (UB) is not the original waveform groundruth (GT), or GT-Encodec32 that has undergone a full encoding-decoding reconstruction by Encodec, but GT-Encodec8 reconstructed from the same 8 codebooks used for training. For comparison, we adopt BSRNN [5], SepFormer [6], TF-LocoFormer [57], and MossFormer2 [58] as baseline methods.

Evaluation Metrics: We conduct both subjective (only for 6 selected systems) and objective tests to evaluate the quality of the SLM’s speech separation. For the objective test, we report speaker similarity, word error rate (WER) by pretrained Whisper [55] base model, Levenshtein Phoneme Similarity (LPS) [59], and SpeechBERTScore (SBS) [60]. In terms of signal quality, we recorded the PESQ [61] (wb) and tested the non-intrusive perceptual metrics UTMOS [62] and DNSMOS [63]. In addition, to further analyze Encodec’s and WavTokenizer’s internal consistency, we compute the token error rate (TER), which is obtained between the predicted zero-order codebook sequence and the ground truth zeroorder codebook. It is important to note that the TERs of the two neural codecs are not comparable. For the comparison models, the main framework’s Encodec is used uniformly for TER computation. For fair evaluation, discriminative baseline outputs are truncated according to the source-active timestamps retained during data simulation, so that all systems are evaluated over consistent active-speech regions. For the subjective test, we conduct a blind multi-stimulus listening test of perceived speech quality with 30 volunteers of different ages, genders, and occupations on 16 single-source samples from 8 mixtures. For each sample, listeners hear all 6 anonymized versions in randomized order before rating the overall perceived quality of each. These comprise 3 SLM-SS configurations, 2 strong discriminative baselines, and the original recording. We limit each trial to 6 conditions because comparing too many versions would make relative quality judgments more difficult.

## B. Results of the Objective Metrics

Our experimental results are summarized in Table I. As shown in the table, SLM-SS achieves a WER of 7.24%, substantially outperforming BSRNN (20.02%) and SepFormer (17.77%), and also surpassing MossFormer2 (8.78%), while

TF-LocoFormer achieves the lowest WER (5.88%). A similar metric-dependent pattern is observed for LPS and SBS: SLM-SS outperforms BSRNN and SepFormer but does not consistently surpass the more recent discriminative baselines. Its speaker similarity is slightly higher than that of TF-LocoFormer but lower than those of BSRNN, SepFormer, and MossFormer2, indicating competitive rather than uniformly superior speaker-identity preservation. Unlike discriminative models with sample-aligned outputs, autoregressive SLMs do not explicitly constrain the generated sequence length to the reference duration. Although wideband PESQ performs delay alignment and zero-padding, it cannot compensate for global time-scale differences; thus, PESQ may also reflect residual temporal drift and duration mismatch rather than waveformreconstruction quality alone.

Introducing the per-talker NAR module consistently improves all reported metrics over the original SLM-SS. Its WER and TER approach those of GT-Encodec8, suggesting that decoupling NAR reconstruction from the SOT separation task improves reconstruction quality. The WavTokenizerbased AR-only model retains comparable speaker similarity and SBS, with moderately higher WER and lower PESQ, but improves UTMOS and DNSMOS over the Encodecbased models. These results suggest that WavTokenizer favors perception-oriented quality without uniformly improving linguistic accuracy or intrusive signal fidelity, reflecting different optimization objectives of the two neural codecs.

To further contextualize the objective results in Table I, we analyze how the distortion introduced by neural codecs affects these metrics. Even when restoring the full 32-order codebooks of Encodec, the reconstructed audio still exhibits noticeable information loss, which becomes more pronounced when only the first 8 orders are retained. This irreversible distortion, introduced by feature discretization, results in slight degradation across objective metrics when compared to the original audio. However, such degradation is less perceptible in subjective listening tests, thereby supporting our hypothesis that discretization-induced artifacts do not always correlate with human auditory perception. A similar effect is reflected in TER: audio reconstructed from the full 32-order codebooks still produces token errors after re-encoding, confirming the persistence of discretization-related distortion. When only the predicted 8-order codebooks are decoded before reencoding, SLM-SS exhibits stronger mismatches; nevertheless, the degradation remains less severe than that observed with SepFormer and BSRNN, benefiting from the Encodec-centered representation framework.

## C. Analysis of Subjective Listening Results

Overall, both SLM-SS and the baseline methods inevitably introduce certain distortions and mismatches. Meanwhile, different evaluation metrics focus on distinct aspects of speech quality, which may lead to inconsistent trends across metrics. To more accurately reflect their practical impact, greater emphasis should therefore be placed on subjective listening tests. From this perspective, the audio reconstructed by the per-talker NAR SLM-SS remains largely comparable to the original recording and falls between the two advanced discriminative models. Furthermore, thanks to WavTokenizer’s targeted optimizations for signal quality and human auditory perception, the WavTokenizer-based approach achieved the highest mean subjective score. Pairwise Wilcoxon signed-rank tests further show that the improvement of SLM-SS-WT over the other evaluated systems is statistically significant after Holm correction. These results indicate that codec-induced distortions do not necessarily translate into a comparable degradation in perceived speech quality.

TABLE I  
OVERALL COMPARISON OF SLM-SS WITH EXISTING APPROACHES.
<table><tr><td rowspan="2">Method</td><td colspan="5">Speaker &amp; Linguistic Consistency</td><td>Signal</td><td colspan="2">Non-intrusive</td><td rowspan="2">Subjective Score↑</td></tr><tr><td>Spk Sim.↑</td><td>WER↓</td><td>TER↓</td><td>LPS↑</td><td>SBS↑</td><td>PESQ↑</td><td>UTMOS↑</td><td>DNSMOS↑</td></tr><tr><td>GT</td><td></td><td>5.19</td><td></td><td>1.000</td><td>1.000</td><td>4.64</td><td>4.09</td><td>3.59</td><td>78.41±4.08</td></tr><tr><td>GT-Encodec32</td><td>93.5</td><td>6.03</td><td>24.7</td><td>0.975</td><td>0.957</td><td>3.14</td><td>3.18</td><td>3.50</td><td></td></tr><tr><td>GT-Encodec8 (UB)</td><td>92.8</td><td>6.31</td><td>39.0</td><td>0.970</td><td>0.944</td><td>2.62</td><td>2.95</td><td>3.38</td><td></td></tr><tr><td>GT-WavTokenizer</td><td>92.7</td><td>7.58</td><td>*47.3</td><td>0.964</td><td>0.939</td><td>2.33</td><td>4.01</td><td>3.67</td><td></td></tr><tr><td>BSRNN</td><td>92.9</td><td>20.02</td><td>65.1</td><td>0.916</td><td>0.903</td><td>2.37</td><td>2.96</td><td>3.02</td><td></td></tr><tr><td>SepFormer</td><td>93.4</td><td>17.77</td><td>70.0</td><td>0.949</td><td>0.910</td><td>2.78</td><td>3.21</td><td>3.15</td><td></td></tr><tr><td>TF-LocoFormer</td><td>91.6</td><td>5.88</td><td>68.3</td><td>0.973</td><td>0.919</td><td>3.51</td><td>3.79</td><td>3.56</td><td>79.23±3.84</td></tr><tr><td>MossFormer2</td><td>92.2</td><td>8.78</td><td>62.6</td><td>0.944</td><td>0.916</td><td>2.95</td><td>3.27</td><td>3.24</td><td>71.59±4.56</td></tr><tr><td>SLM-SS</td><td>91.7</td><td>7.24</td><td>45.8</td><td>0.954</td><td>0.913</td><td>2.17</td><td>2.64</td><td>3.21</td><td>74.58±4.21</td></tr><tr><td>+ Per-talker NAR</td><td>92.3</td><td>6.92</td><td>41.7</td><td>0.959</td><td>0.925</td><td>2.30</td><td>2.88</td><td>3.36</td><td>76.83±4.22</td></tr><tr><td>+ WavTokenizer</td><td>92.3</td><td>8.97</td><td>*66.8</td><td>0.923</td><td>0.921</td><td>2.06</td><td>3.95</td><td>3.65</td><td>81.41±3.69</td></tr></table>

TER values marked with <sup>∗</sup> are computed using WavTokenizer. They are not directly comparable with others using Encodec. Subjective scores are reported as mean ± 95% confidence interval. We limit each subjective listening trial to 6 conditions because comparing too many versions would make quality judgments difficult for listeners.

To more clearly compare the differences in the audio generated by the two paradigms, we selected a few samples for visualization, as shown in Fig. 4. It can be observed that truncating Encodec to only the first 8 codebooks (GT8) leads to the loss of high-frequency components, resulting in a degradation in the clarity of the mel-spectrogram. However, this compression does not cause significant formant smearing, and the harmonic structure remains largely distinguishable. Since the proposed SLM-SS framework adopts GT-Encodec8 (GT8) as an upper performance bound, its spectral characteristics are largely consistent with those of GT-Encodec8.

In contrast, SepFormer operates directly on the waveform domain and thus preserves more detailed spectral information, exhibiting less degradation in mel-spectrogram resolution. Nevertheless, under challenging conditions such as severe source overlap or highly imbalanced talker energy, SepFormer tends to introduce over-suppression artifacts, which may lead to the attenuation or loss of certain speech segments.

## D. Ablation Study on Temperature During Inference

In this experiment, we investigate whether the proposed method requires temperature tuning during inference. The motivation of this ablation is threefold. First, in autoregressive generation, temperature directly controls token uncertainty and therefore affects reconstruction quality. Second, it influences decoding stability: excessively low temperature often leads to over-confident but rigid token selection, while excessively high temperature can cause unstable token trajectories and error accumulation. Third, it is closely related to hallucination behavior in codec-token generation (e.g., speaker confusion, non-speech artifacts, or content drift). This concern is also consistent with prior neural codec language modeling practice, including VALL-E [22], where temperature is commonly tuned at inference to balance fidelity and diversity. Therefore, we conduct a systematic ablation over temperature to verify whether SLM-SS requires such tuning and to identify a robust operating point. As reported in Table II, the proposed method achieves optimal results using the default temperature setting of 1.0, without requiring additional tuning. This result suggests that our model can maintain good generation quality and stable decoding with minimal risk of temperature-induced hallucination, without additional inference-time tuning.

TABLE II  
PERFORMANCE ON DIFFERENT AR TEMPERATURES.
<table><tr><td>Temperature</td><td>Spk Sim.↑</td><td>WER↓</td><td>TER↓</td><td>LPS↑</td><td>SBS↑</td></tr><tr><td>0.5</td><td>38.9</td><td>49.08</td><td>69.3</td><td>0.581</td><td>0.695</td></tr><tr><td>0.9</td><td>73.1</td><td>10.20</td><td>56.9</td><td>0.900</td><td>0.845</td></tr><tr><td>1.0</td><td>91.7</td><td>7.24</td><td>45.8</td><td>0.954</td><td>0.913</td></tr><tr><td>1.1</td><td>77.8</td><td>9.73</td><td>52.0</td><td>0.949</td><td>0.895</td></tr><tr><td>1.5</td><td>54.2</td><td>64.55</td><td>87.8</td><td>0.178</td><td>0.497</td></tr></table>

## E. Ablation Study on Number of Codebooks

As discussed earlier, the proposed method is inherently constrained by the number of codebooks employed for modeling. As shown in Table I, restoration quality is strongly correlated with the number of codebooks, where using fewer codebooks results in increasingly severe reconstruction distortion. Theoretically, optimal performance would be achieved by fully leveraging Encodec’s complete 32-order codebook sequence. However, due to the use of independent token embedding layers for different orders in the NAR model, predicting higher-order codebooks requires jointly integrating information from all lower-order codebooks. This design substantially increases training complexity and computational cost, leading to significantly slower convergence in practice. After carefully balancing training efficiency, computational overhead, and reconstruction quality, we finally select the top 8 codebooks for modeling in our main experiments. This choice represents a practical compromise that preserves most of the perceptual and linguistic benefits while maintaining a feasible training cost.

![](images/01a04507edc44402e52d0c15ab5df14e98430a8adb087dc6ca9312c2dc24436d.jpg)  
Fig. 4. Comparison of time series and mel-frequency spectrograms across different models. Truncating Encodec to the first 8 codebooks (GT8) removes high frequency details and slightly degrades mel-spectrogram clarity while preserving formant structure, leading SLM-SS to exhibit similar spectral characteristics. In contrast, SepFormer retains finer spectral resolution by directly operating on waveforms, but may introduce over-suppression artifacts under challenging conditions such as heavy overlap or imbalanced talker energy, as marked areas 1, 2, and 3.

![](images/ccc2ad13595806e93f4f65b3ca108901d4a7cea9aef8ac0055c50b816550f8da.jpg)  
Fig. 5. Performance variation of WER and LPS with the number of codebooks. The codebook exhibits distinct residual characteristics, and its performance changes nonlinearly; when the number of codebooks exceeds 6, the improvement in performance gradually slows down.

To quantitatively analyze this relationship, we evaluate multiple metrics using the first 1 to 7 codebooks and illustrate the corresponding performance trends in Fig. 5. As the number of codebooks increases, model performance exhibits a clear and consistent positive correlation across all evaluated metrics. These results indicate that incorporating additional codebooks can further enhance the overall performance of our method.

## F. Performance on Scenarios with Dynamic Numbers of Talkers

We further investigate the performance of the SLM-SS framework under scenarios involving different numbers of talkers. Specifically, we train and evaluate the model on Libri2Mix, Libri3Mix, and their joint training set to assess robustness to varying numbers of speakers. Due to computational constraints in this experiment, we replace the Whispermedium configuration with Whisper-base, while keeping all other training hyperparameters unchanged. This experimental setting enables a controlled comparison across different talker scenarios without introducing additional confounding factors. The corresponding results are reported in Table III.

Table III reveals a clear talker-count effect: compared with 2Mix, performance degrades consistently in the 3Mix-only setting across all objective metrics, confirming that threetalker separation is substantially more challenging. This trend is expected because higher-talker mixtures produce longer SOT sequences, which increase autoregressive modeling difficulty. More importantly, joint training on 2Mix and 3Mix improves robustness in the harder 3Mix condition, with a clear WER reduction (12.1%→ 9.9%) and consistent gains on the remaining metrics, while preserving performance in the easier 2Mix condition. Overall, mixed-talker training provides a better robustness-accuracy trade-off than training on a single talker-count setting.

TABLE III  
PERFORMANCE WITH DIFFERENT NUMBERS OF TALKERS.
<table><tr><td>Train</td><td>Test</td><td>Spk sim</td><td>WER</td><td>TER</td><td>LPS</td><td>SBS</td></tr><tr><td>2Mix</td><td>2Mix</td><td>90.2</td><td>8.84</td><td>51.1</td><td>0.936</td><td>0.900</td></tr><tr><td>3Mix</td><td>3Mix</td><td>87.3</td><td>12.12</td><td>63.7</td><td>0.895</td><td>0.870</td></tr><tr><td rowspan="2">Joint</td><td>2Mix</td><td>90.2</td><td>8.80</td><td>50.7</td><td>0.939</td><td>0.903</td></tr><tr><td>3Mix</td><td>88.8</td><td>9.94</td><td>61.3</td><td>0.915</td><td>0.875</td></tr></table>

## G. Robustness Under Increasing Number of Talker

To evaluate the scalability of the SLM-SS to more challenging mixtures, we extend the LibriMix data simulation pipeline to construct a 4-talker setting. With all hyperparameters kept unchanged, we generate a Libri4Mix dataset (with a smallscale train set and a test set) and finetune the model initialized from AR weights jointly trained on 2 and 3 talker scenarios.

The model converges within three epochs under the limited 4-talker data, after which the validation loss increases, indicating overfitting. This suggests that the framework can be effectively adapted to higher talker-count scenarios via lightweight finetuning, even under low-resource conditions. We further assess talker number prediction across different scenarios by applying the fine-tuned 4-talker model to all settings. The resulting confusion matrices are shown in Fig. 6.

![](images/7e8c7b2bc0f769538b98672701c29c521b9912c46a84dbfbe6b70a71586cf406.jpg)  
Fig. 6. Confusion matrix for talker number prediction accuracy across different scenarios.

The performance on 2- and 3-talker scenarios remains largely unaffected after finetuning, with only a small number of prediction errors, due to early stopping preventing severe overfitting. In contrast, performance in the 4-talker scenario is lower due to increased task difficulty and limited data, with occasional underestimation of talker counts. Nevertheless, the model still demonstrates strong overall separation capability, highlighting the robustness of the proposed framework in more challenging multi-talker settings.

Furthermore, we adopt the per-talker NAR approach for high-order sequence decoding to evaluate the quality of the final audio output. The NAR model, jointly trained on 2 and 3 talker scenarios, is finetuned using the same strategy as the AR component and evaluated across different talker settings. Additionally, to investigate whether the per-talker NAR model possesses the generalization capability to directly apply to four-talker scenarios, we also conducted direct tests using the unfinetuned weights.

TABLE IV  
PERFORMANCE COMPARISON ACROSS DIFFERENT NUMBERS OF TALKERS AFTER EXTENSION
<table><tr><td>NAR</td><td>#Talker</td><td>Spk Sim.↑</td><td>WER↓</td><td>TER↓</td><td>LPS↑</td><td>SBS↑</td></tr><tr><td rowspan="3">Unfinetuned</td><td>2Mix</td><td>90.1</td><td>8.83</td><td>50.5</td><td>0.937</td><td>0.903</td></tr><tr><td>3Mix</td><td>88.7</td><td>9.93</td><td>61.3</td><td>0.912</td><td>0.871</td></tr><tr><td>4Mix</td><td>79.5</td><td>16.19</td><td>70.1</td><td>0.833</td><td>0.794</td></tr><tr><td rowspan="3">Finetuned</td><td>2Mix</td><td>90.1</td><td>8.83</td><td>50.6</td><td>0.938</td><td>0.903</td></tr><tr><td>3Mix</td><td>88.7</td><td>9.92</td><td>61.4</td><td>0.911</td><td>0.872</td></tr><tr><td>4Mix</td><td>84.6</td><td>11.10</td><td>66.7</td><td>0.889</td><td>0.852</td></tr></table>

As shown in Table IV, when the NAR model is not finetuned on the 4-talker data, performance in the 2 and 3 talker scenarios is largely governed by the AR component finetuned for the 4-talker setting, with no significant performance changes observed. However, in the 4-talker scenario, the original pertalker NAR encoder lacks the capacity to model more complex mixtures, leading to insufficient extraction of higher-order codebooks and consequent performance degradation, but it still offers a certain level of performance.

In contrast, after finetuning the per-talker NAR model, performance in the 2 and 3 talker scenarios remains stable, while performance in the 4-talker scenario improves substantially. These results indicate a strong correlation between the number of talkers (i.e., inference difficulty) and model performance, demonstrating that the SLM-SS framework effectively supports variable numbers of talkers and maintains consistent performance across different speaker configurations.

## H. Results on AR-Only Model

To study the effect of the neural codec choice, we replace Encodec with WavTokenizer while keeping the AR backbone and training recipe unchanged. Concretely, we only adjust the vocabulary size from 1024 to 4096 to match WavTokenizer; all other hyperparameters remain identical to ensure a fair comparison. The AR model is trained separately on 2- and 3-talker mixtures.

As summarized in the last row of Table I, switching from Encodec to WavTokenizer yields comparable separation quality (e.g., SBS) while slightly degrading WER. We attribute this trend to the codec objective: WavTokenizer is more aggressively optimized at the signal level, which benefits perceptual quality and stability, whereas it is not explicitly tailored for ASR-related objectives.

Beyond the metric comparison, WavTokenizer also changes the system design trade-off. In our framework, it removes the need for the additional per-talker NAR module for high-order codebook decoding, which simplifies the training/inference pipeline and reduces the overall compute cost.

## I. Decoding Strategies with AR-Only Model

We further analyze the impact of autoregressive decoding under WavTokenizer in Table V. We consider greedy decoding, beam search $( b e a m s i z e = 2 0 )$ , and a lightweight stochastic variant, Top-30 Random Sampling (Top-30 RS): at each step, we first select the 30 tokens with the highest logits and then uniformly sample one token from this candidate set.

TABLE V  
PERFORMANCE USING WAVTOKENIZER UNDER DIFFERENT TALKER SCENARIOS AND DECODING STRATEGIES.
<table><tr><td>Dataset</td><td>Decode method</td><td>Spk Sim.↑</td><td>WER↓</td><td>LPS ↑</td><td>SBS↑</td></tr><tr><td rowspan="3">2Mix</td><td>Greedy Search</td><td>91.9</td><td>9.66</td><td>0.904</td><td>0.915</td></tr><tr><td>Top-30 RS</td><td>92.0</td><td>9.08</td><td>0.917</td><td>0.919</td></tr><tr><td>Beam Search</td><td>92.3</td><td>8.97</td><td>0.923</td><td>0.921</td></tr><tr><td rowspan="3">3Mix</td><td>Greedy Search</td><td>87.2</td><td>11.41</td><td>0.906</td><td>0.910</td></tr><tr><td>Top-30 RS</td><td>87.8</td><td>11.03</td><td>0.908</td><td>0.911</td></tr><tr><td>Beam Search</td><td>88.8</td><td>10.18</td><td>0.915</td><td>0.913</td></tr></table>

Compared with Top-1 greedy decoding, Top-30 RS provides a more robust decoding behavior (consistent gains on both 2Mix and 3Mix), likely because the larger and more redundant WavTokenizer vocabulary makes the model less sensitive to occasional over-confident token choices. Beam search still delivers the best overall performance, indicating the expected trade-off between decoding efficiency and optimality.

Overall, the codec choice affects both the achievable performance and the computational profile of SLM-SS. Exploring more advanced neural codecs (and their matching decoding strategies) remains a promising direction for future work.

## J. Ablation Study on Number of Model Parameters

Since the codebook sequence length increases linearly with the number of talkers, SLM require sufficient capacity for autoregressive decoding. Insufficient capacity may cause cumulative errors over long sequences, whereas excessive capacity substantially increases computational overhead.

To investigate the trade-off between model size and performance, we evaluated two pre-trained encoder sizes and four scratch-built decoder sizes for SLM-SS with the WavTokenizer-based approach. We report performance, parameter count, computational cost (GMACs) at 16 kHz, peak memory allocation, and Real-time Factor (RTF), and compare them with two advanced discriminative models. GMACs are computed per second of 16-kHz input audio. RTF and peak allocated GPU memory are measured on an RTX 4090 using FP16 inference with batch size 1. The reported RTF includes autoregressive beam-search decoding with beamsize = 20.

As shown in Table VI, the encoder extracts latent representations from multi-talker audio and guides the autoregressive decoder through cross-attention. Insufficient encoder capacity therefore limits decoding performance and cannot be fully compensated for by a larger decoder. With a sufficiently powerful encoder, a moderately sized decoder achieves strong performance, while an overly small decoder may struggle with long sequences. Further increasing decoder size yields diminishing gains with substantially higher computational overhead.

Despite having fewer parameters, the computational costs of discriminative models remain high and are comparable to those of our medium-sized SLM model. They directly predict masks or spectrum mappings without autoregressive decoding, achieving much lower RTF than all SLM configurations. This holds even when some discriminative models requires higher overall computational overhead than SLM models. Due to the non-causal encoder and full-context cross-attention, the current SLM-SS implementation is intended for offline inference.

## K. Performance on Out-of-domain Data

To further investigate the generalization ability of the SLM-SS model, we selected two out-of-domain datasets: the Libri2Mix noisy testset, which incorporates WHAM! [64] noise; and the AISHELL1-2Mix Chinese noise-free dataset, generated based on AISHELL1 [65] using a simulation pipeline similar to the Librimix. We selected models from the l-l and l-s configurations in Table VI, both of which were trained on the clean LibriMix dataset, to evaluate their crosslingual and noise-robust capabilities. For comparison, we also tested TF-LocoFormer and MossFormer2, which were trained on the same LibriMix dataset.

As shown in Table VII, SLM-SS achieves the best DNS-MOS and UTMOS on both out-of-domain datasets. In terms of WER, the l-l configuration outperforms the discriminative models on LibriMix noisy, while performing slightly worse than TF-LocoFormer on the Chinese dataset, demonstrating the cross-language generalization and noise robustness of the SLM models.

## V. DISCUSSION

## A. Neural codecs are the dominant performance bottleneck.

Our experiments indicate that, in Encodec-based AR–NAR separation, the effective performance ceiling is the codec reconstruction from the modeled codebooks (GT-Encodec8), rather than the original waveform. Consequently, codec choice and how its representation is exploited largely determine the attainable separation quality. Under residual vector quantization, adding higher-order codebooks helps (Fig. 5) but substantially complicates optimization, since each higher-order prediction must condition on an expanding set of lowerorder sequences—highlighting a trade-off between information completeness and trainability/efficiency.

WavTokenizer further shows that codec design is a firstorder algorithmic decision: a single-order codec can remove the NAR refinement stage, simplifying training/inference and improving subjective quality. Yet its objective (often waveform/perception oriented) may mismatch ASR-centric intelligibility metrics, yielding different WER–MOS trade-offs. Overall, progress may depend more on co-designing the codec and token-prediction objective for separation and intelligibility than on scaling the SLM backbone alone.

TABLE VI  
MODEL SCALING AND COMPUTATIONAL EFFICIENCY COMPARISON.
<table><tr><td>Configuration</td><td>#Params</td><td>GMACs</td><td>Mem</td><td>RTF</td><td>WER↓</td><td>PESQ↑</td><td>UTMOS↑</td><td>DNSMOS↑</td></tr><tr><td>WavLM Base+ Encoder (94M)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ 23M Decoder (b-t)</td><td>117M</td><td>37.14</td><td>0.72GiB</td><td>0.29</td><td>72.87</td><td>1.22</td><td>2.21</td><td>3.07</td></tr><tr><td>+ 48M Decoder (b-s)</td><td>142M</td><td>69.52</td><td>1.44GiB</td><td>0.52</td><td>63.88</td><td>1.34</td><td>2.58</td><td>3.16</td></tr><tr><td>+ 121M Decoder (b-m)</td><td>215M</td><td>161.01</td><td>2.18GiB</td><td>0.75</td><td>36.91</td><td>1.65</td><td>3.01</td><td>3.29</td></tr><tr><td>+ 212M Decoder (b-1)</td><td>306M</td><td>348.73</td><td>2.90GiB</td><td>0.91</td><td>25.22</td><td>1.91</td><td>3.32</td><td>3.40</td></tr><tr><td>WavLM Large Encoder (315M)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ 23M Decoder (l-t)</td><td>338M</td><td>45.82</td><td>1.47GiB</td><td>0.37</td><td>46.29</td><td>1.52</td><td>2.40</td><td>3.11</td></tr><tr><td>+ 48M Decoder (l-s)</td><td>363M</td><td>84.56</td><td>2.19GiB</td><td>0.64</td><td>9.43</td><td>2.07</td><td>3.97</td><td>3.61</td></tr><tr><td>+ 121M Decoder (l-m)</td><td>436M</td><td>177.83</td><td>2.93GiB</td><td>0.93</td><td>10.14</td><td>2.06</td><td>3.94</td><td>3.66</td></tr><tr><td>+ 212M Decoder (l-1, ours)</td><td>527M</td><td>366.39</td><td>3.66GiB</td><td>1.35</td><td>8.97</td><td>2.06</td><td>3.95</td><td>3.65</td></tr><tr><td>TF-LocoFormer</td><td>15M</td><td>250.20</td><td>1.19GiB</td><td>0.08</td><td>5.88</td><td>3.51</td><td>3.79</td><td>3.56</td></tr><tr><td>MossFormer2</td><td>53M</td><td>105.06</td><td>1.44GiB</td><td>0.07</td><td>8.78</td><td>2.95</td><td>3.27</td><td>3.24</td></tr></table>

TABLE VII  
OUT-OF-DOMAIN EVALUATION ON LIBRI2MIX-NOISY AND AISHELL1-2MIX.
<table><tr><td>Model</td><td>WER/CER↓</td><td>PESQ↑</td><td>UTMOS↑</td><td>DNSMOS↑</td></tr><tr><td colspan="5">(a) Libri2Mix-noisy</td></tr><tr><td>SLM-SS-WT (l-s)</td><td>30.46</td><td>1.60</td><td>2.96</td><td>3.20</td></tr><tr><td>SLM-SS-WT (1-1)</td><td>25.54</td><td>1.61</td><td>2.98</td><td>3.22</td></tr><tr><td>TF-LocoFormer</td><td>25.96</td><td>1.64</td><td>2.11</td><td>2.56</td></tr><tr><td>MossFormer2</td><td>33.89</td><td>1.39</td><td>1.54</td><td>2.35</td></tr><tr><td colspan="5">(b) AISHELL1-2Mix</td></tr><tr><td>SLM-SS-WT (l-s)</td><td>38.35*</td><td>1.56</td><td>2.65</td><td>3.48</td></tr><tr><td>SLM-SS-WT (1-1)</td><td>33.70*</td><td>1.55</td><td>2.66</td><td>3.48</td></tr><tr><td>TF-LocoFormer</td><td>32.58*</td><td>2.25</td><td>1.82</td><td>2.91</td></tr><tr><td>MossFormer2</td><td>47.36*</td><td>1.77</td><td>1.46</td><td>2.90</td></tr></table>

WER is reported for Libri2Mix-noisy, while CER is reported for AISHELL1-2Mix.

B. Why use continuous mixture inputs with discrete token outputs?

We adopt a continuous encoder with a discrete token decoder because current neural codecs are predominantly trained on single-speaker speech, and thus are not guaranteed to produce a well-behaved discrete representation when directly encoding multi-talker mixtures. If we were to feed mixture waveforms into a codec and then run a purely decoder-only token model, the discretization step itself could become a major failure point: mixture interference may be absorbed into the codec tokens in an entangled manner, making it difficult to recover speaker-consistent sequences.

By keeping the mixture processing in a continuous encoder, the attention-based encoder-decoder model can learn separation-relevant acoustic representations directly from the mixture, while using discrete tokens only on the output side as an intelligibility-oriented generative target. This hybrid design also keeps the framework compatible with future codecs: if a high-quality multi-talker codec (or a mixtureinvariant, speaker-disentangling tokenizer) becomes available, our framework could be naturally simplified into a decoderonly architecture that predicts tokens conditioned on discretized mixture inputs, further reducing system complexity and latency.

## C. Limitations and future work

Codec–model co-design. The strongest path forward is likely to adopt codecs with fewer codebooks but higher information density, or to learn task-specific tokenizers that better preserve intelligibility-relevant cues under aggressive compression. Jointly training (or at least jointly selecting) codec representations and separation objectives may reduce the current gap between perceptual quality and downstream intelligibility. Scaling to more talkers. Higher talker counts lengthen the SOT sequence and increase uncertainty in speakerchange/boundary decisions, making AR decoding more sensi tive to early token errors (e.g., boundary shifts, speaker swaps, or incorrect talker counts). While finetuning can extend the framework (e.g., to 4 talkers), performance is quickly limited by data scarcity at higher talker counts; addressing this likely requires more diverse high-overlap mixtures and more robust decoding constraints or auxiliary objectives (e.g., on <SC> placement) to stabilize boundary prediction.

Real-world robustness. LibriMix is simulated and largely clean; it does not fully cover realistic reverberation, background noise, channel mismatch, and spontaneous speech. Extending SLM-SS to such conditions likely requires either domain-augmented training or explicit modeling of acoustics/noise in the token space.

## VI. CONCLUSION

We presented SLM-SS, a token-based generative speech separation framework that leverages speech language models. Clean targets are discretized into multi-codebook sequences with neural codecs and serialized with SOT, turning multi-talker separation into sequence generation. An encoder– decoder AR model predicts the zero-order codebook to establish speaker ordering and coarse linguistic content, and a lightweight per-talker NAR module then refines higher-order residual codebooks for efficient reconstruction.

Experiments show that per-talker refinement improves recognition accuracy, while the single-codebook WavTokenizer variant simplifies generation and improves listener-rated quality. Its non-intrusive quality advantages over the tested baselines persist on noisy and Chinese mixtures. Together, these results demonstrate SLM-SS’s ability to recover intelligible, perceptually high-quality speech from multi-talker mixtures through codec-token generation. Future work will focus on codec–model co-design, faster decoding, and robustness to more challenging multi-talker conditions.

## REFERENCES

[1] DeLiang Wang and Jitong Chen, “Supervised speech separation based on deep learning: An overview,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 10, pp. 1702–1726, 2018.

[2] John R. Hershey, Zhuo Chen, Jonathan Le Roux, and Shinji Watanabe, “Deep clustering: Discriminative embeddings for segmentation and separation,” in 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016, pp. 31–35.

[3] Yuzhou Liu and DeLiang Wang, “Divide and conquer: A deep casa approach to talker-independent monaural speaker separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 12, pp. 2092–2102, 2019.

[4] Yi Luo, “Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 8, pp. 1256–1266, 2019.

[5] Yi Luo and Jianwei Yu, “Music source separation with band-split rnn,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 1893–1901, 2023.

[6] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and Jianyuan Zhong, “Attention is all you need in speech separation,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 21–25.

[7] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, and John R. Hershey, “Sdr – half-baked or well done?,” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 626–630.

[8] Kazuma Iwamoto, Tsubasa Ochiai, Marc Delcroix, Rintaro Ikeshita, Hiroshi Sato, Shoko Araki, and Shigeru Katagiri, “How bad are artifacts?: Analyzing the impact of speech enhancement errors on ASR,” in Interspeech 2022, 2022, pp. 5418–5422.

[9] Bo Chen, Chao Wu, and Wenbin Zhao, “Sepdiff: Speech separation based on denoising diffusion model,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[10] Robin Scheibler, Youna Ji, Soo-Whan Chung, Jaeuk Byun, Soyeon Choe, and Min-Seok Choi, “Diffusion-based generative speech source separation,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[11] Robin Scheibler, John R. Hershey, Arnaud Doucet, and Henry Li, “Source Separation by Flow Matching,” in 2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), Oct. 2025, pp. 1–5.

[12] Jiahe Wang, Hongyu Wang, Wei Wang, Lei Yang, Chenda Li, Wangyou Zhang, Lufen Tan, and Yanmin Qian, “MeanSE: Efficient Generative Speech Enhancement with Mean Flows,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2026, pp. 16827–16831.

[13] Sumukh K Aithal, Pratyush Maini, Zachary C. Lipton, and J. Zico Kolter, “Understanding hallucinations in diffusion models through mode interpolation,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds. 2024, vol. 37, pp. 134614–134644, Curran Associates, Inc.

[14] Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Steven Y. Guo, and Irwin King, “Recent advances in speech language models: A survey,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, Eds., Vienna, Austria, July 2025, pp. 13943– 13970, Association for Computational Linguistics.

[15] Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, zehan wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, YIDI JIANG, Qian Chen, Siqi Zheng, and Zhou Zhao, “Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling,” in International Conference on Representation Learning, 2025, vol. 2025, pp. 93809–93826.

[16] Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi,´ “High fidelity neural audio compression,” Transactions on Machine Learning Research, 2023, Featured Certification, Reproducibility Certification.

[17] Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar, “High-fidelity audio compression with improved rvqgan,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds. 2023, vol. 36, pp. 27980–27993, Curran Associates, Inc.

[18] Jiankai Huang, Junteng Zhang, Ming Lu, Xun Cao, and Zhan Ma, “Lisa: Lightweight yet superb neural speech coding,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 14457–14461.

[19] R. Gray, “Vector quantization,” IEEE ASSP Magazine, vol. 1, no. 2, pp. 4–29, Apr. 1984.

[20] A. Vasuki and P.T. Vanathi, “A review of vector quantization techniques,” IEEE Potentials, vol. 25, no. 4, pp. 39–47, July 2006.

[21] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin, “Qwen3-ASR Technical Report,” Jan. 2026.

[22] Sanyuan Chen, Chengyi Wang, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, and Furu Wei, “Neural codec language models are zeroshot text to speech synthesizers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 705–718, 2025.

[23] Yue Zhou, Peng Bai, and Xiaodong Shi, “UniVoice: A unified framework for text-to-speech, singing voice synthesis, and opera singing synthesis,” Applied Intelligence, vol. 56, no. 4, pp. 104, Feb. 2026.

[24] Luca A. Lanzendorfer, Fr ¨ ed´ eric Berdoz, Antonis Asonitis, and Roger´ Wattenhofer, “High-Fidelity Speech Enhancement Via Discrete Audio Tokens,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2026, pp. 17397–17401.

[25] Beilong Tang, Bang Zeng, and Ming Li, “TSELM: Target Speaker Extraction Using Discrete Tokens and Language Models,” in Man-Machine Speech Communication, Jia Jia, Zhiyong Wu, Lijian Gao, Gongping Huang, and Ya Li, Eds., Singapore, 2026, pp. 459–469, Springer Nature.

[26] Tianhua Li, Chenda Li, Wei Wang, Xin Zhou, Xihui Chen, Jianqing Gao, and Yanmin Qian, “SLM-SS: Speech Language Model for Generative Speech Separation,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2026, pp. 15432–15436.

[27] Naoyuki Kanda, Yashesh Gaur, Xiaofei Wang, Zhong Meng, and Takuya Yoshioka, “Serialized output training for end-to-end overlapped speech recognition,” in Interspeech 2020, 2020, pp. 2797–2801.

[28] Chenda Li, Yao Qian, Zhuo Chen, Naoyuki Kanda, Dongmei Wang, Takuya Yoshioka, Yanmin Qian, and Michael Zeng, “Adapting multilingual asr models for handling multiple talkers,” in Interspeech 2023, 2023, pp. 1314–1318.

[29] Jiawen Kang, Lingwei Meng, Mingyu Cui, Haohan Guo, Xixin Wu, Xunying Liu, and Helen Meng, “Cross-speaker encoding network for multi-talker speech recognition,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 11986–11990.

[30] Zhong-Qiu Wang, Peidong Wang, and DeLiang Wang, “Complex Spectral Mapping for Single- and Multi-Channel Speech Enhancement and Robust ASR,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1778–1787, 2020.

[31] Dong Yu, Morten Kolbæk, Zheng-Hua Tan, and Jesper Jensen, “Permutation invariant training of deep models for speaker-independent multitalker speech separation,” in 2017 IEEE International Conference on

Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017, pp. 241–245.

[32] Morten Kolbæk, Dong Yu, Zheng-Hua Tan, and Jesper Jensen, “Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 10, pp. 1901– 1913, 2017.

[33] Yi Luo and Nima Mesgarani, “Tasnet: Time-domain audio separation network for real-time, single-channel speech separation,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2018, p. 696–700, IEEE Press.

[34] Y. Luo, Z. Chen, and T. Yoshioka, “Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, May 2020, pp. 46– 50.

[35] Jingjing Chen, Qirong Mao, and Dong Liu, “Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation,” in Interspeech 2020. Oct. 2020, pp. 2642–2646, ISCA.

[36] Chenda Li, Lei Yang, Weiqin Wang, and Yanmin Qian, “Skim: Skipping Memory Lstm for Low-Latency Real-Time Continuous Speech Separation,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, May 2022, pp. 681–685.

[37] Kohei Saijo, Janek Ebbers, Franc¸ois G. Germain, Gordon Wichern, and Jonathan Le Roux, “Task-aware unified source separation,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[38] Yoshiki Masuyama, Kohei Saijo, Francesco Paissan, Jiangyu Han, Marc Delcroix, Ryo Aihara, Franc¸ois G. Germain, Gordon Wichern, and Jonathan Le Roux, “Flexio: Flexible single- and multi-channel speech separation and enhancement,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 14417–14421.

[39] Younglo Lee, Shukjae Choi, Byeong-Yeol Kim, Zhong-Qiu Wang, and Shinji Watanabe, “Boosting unknown-number speaker separation with transformer decoder-based attractor,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 446–450.

[40] Srikanth Raj Chetupalli and Emanuel A. P. Habets, “Speaker count-¨ ing and separation from single-channel noisy mixtures,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 1681–1692, 2023.

[41] Kohei Saijo, Wangyou Zhang, Zhong-Qiu Wang, Shinji Watanabe, Tetsunori Kobayashi, and Tetsuji Ogawa, “A single speech enhancement model unifying dereverberation, denoising, speaker counting, separation, and extraction,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2023, pp. 1–6.

[42] Shota Horiguchi, Yusuke Fujita, Shinji Watanabe, Yawen Xue, and Paola Garcia, “Encoder-decoder based attractors for end-to-end neural diarization,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 1493–1507, 2022.

[43] Hongyu Wang, Chenda Li, Xin Zhou, Shuai Wang, and Yanmin Qian, “USE: A Unified Model for Universal Sound Separation and Extraction,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 39, pp. 33476–33484, Mar. 2026.

[44] Yutong Wen, Ke Chen, Prem Seetharaman, Oriol Nieto, Jiaqi Su, Rithesh Kumar, Minje Kim, Paris Smaragdis, Zeyu Jin, and Justin Salamon, “Promptsep: Generative audio separation via multimodal prompting,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 14422–14426.

[45] ByteDance Seed Team, “Seed-tts: A family of high-quality versatile speech generation models,” arXiv preprint arXiv:2406.02430, 2024.

[46] Ye Bai, Jingping Chen, Jitong Chen, Wei Chen, Zhuo Chen, Chuang Ding, Linhao Dong, Qianqian Dong, Yujiao Du, Kepan Gao, et al., “Seed-asr: Understanding diverse speech and contexts with llm-based speech recognition,” arXiv preprint arXiv:2407.04675, 2024.

[47] Kai-Tuo Xu, Feng-Long Xie, Xu Tang, and Yao Hu, “Fireredasr: Open-source industrial-grade mandarin speech recognition models from encoder-decoder to llm integration,” arXiv preprint arXiv:2501.14350, 2025.

[48] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[49] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2022.

[50] Yuanyuan Wang, Hangting Chen, Dongchao Yang, Weiqin Li, Dan Luo, Guangzhi Li, Shan Yang, Zhiyong Wu, Helen Meng, and Xixin Wu, “Unisep: Universal target audio separation with language models at scale,” in 2025 IEEE International Conference on Multimedia and Expo (ICME), 2025, pp. 1–6.

[51] Tanvir Mahmud and Diana Marculescu, “OpenSep: Leveraging large language models with textual inversion for open world audio separation,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, Eds., Miami, Florida, USA, Nov. 2024, pp. 13244– 13260, Association for Computational Linguistics.

[52] Zhaoxi Mu, Xinyu Yang, and Gang Wang, “Sepalm: Audio language models are error correctors for robust speech separation,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, James Kwok, Ed. 8 2025, pp. 8204–8212, International Joint Conferences on Artificial Intelligence Organization, Main Track.

[53] Ziqian Wang, Xinfa Zhu, Zihan Zhang, YuanJun Lv, Ning Jiang, Guoqing Zhao, and Lei Xie, “Selm: Speech enhancement using discrete tokens and language models,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 11561–11565.

[54] Hakan Erdogan, Scott Wisdom, Xuankai Chang, Zalan Borsos, Marco´ Tagliasacchi, Neil Zeghidour, and John R. Hershey, “Tokensplit: Using discrete speech representations for direct, refined, and transcriptconditioned speech separation and recognition,” in Interspeech 2023, 2023, pp. 3462–3466.

[55] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via largescale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28492–28518.

[56] Joris Cosentino, Manuel Pariente, Samuele Cornell, Antoine Deleforge, and Emmanuel Vincent, “Librimix: An open-source dataset for generalizable speech separation,” 2020.

[57] Kohei Saijo, Gordon Wichern, Franc¸ois G. Germain, Zexu Pan, and Jonathan Le Roux, “Tf-locoformer: Transformer with local modeling by convolution for speech separation and enhancement,” in 2024 18th International Workshop on Acoustic Signal Enhancement (IWAENC), 2024, pp. 205–209.

[58] Shengkui Zhao, Yukun Ma, Chongjia Ni, Chong Zhang, Hao Wang, Trung Hieu Nguyen, Kun Zhou, Jia Qi Yip, Dianwen Ng, and Bin Ma, “Mossformer2: Combining transformer and rnn-free recurrent network for enhanced time-domain monaural speech separation,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 10356–10360.

[59] Jan Pirklbauer, Marvin Sach, Kristoff Fluyt, Wouter Tirry, Wafaa Wardah, Sebastian Moeller, and Tim Fingscheidt, “Evaluation metrics for generative speech enhancement methods: Issues and perspectives,” in Speech Communication; 15th ITG Conference, 2023, pp. 265–269.

[60] Takaaki Saeki, Soumi Maiti, Shinnosuke Takamichi, Shinji Watanabe, and Hiroshi Saruwatari, “Speechbertscore: Reference-aware automatic evaluation of speech generation leveraging nlp evaluation metrics,” in Proc. Interspeech 2024, 2024, pp. 4943–4947.

[61] A.W. Rix, J.G. Beerends, M.P. Hollier, and A.P. Hekstra, “Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No.01CH37221), 2001, vol. 2, pp. 749–752 vol.2.

[62] Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari, “UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022,” in Interspeech 2022, 2022, pp. 4521–4525.

[63] Chandan K A Reddy, Vishak Gopal, and Ross Cutler, “Dnsmos: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 6493– 6497.

[64] Gordon Wichern, Joe Antognini, Michael Flynn, Licheng Richard Zhu, Emmett McQuinn, Dwight Crow, Ethan Manilow, and Jonathan Le Roux, “WHAM!: Extending Speech Separation to Noisy Environments,” in Interspeech 2019, 2019, pp. 1368–1372.

[65] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng, “Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline,” in 2017 20th Conference of the Oriental Chapter of the

International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O-COCOSDA), 2017, pp. 1–5.

## VII. AUTHOR BIOGRAPHY

![](images/8fe67a80368360b578c6ed7244755ce88fecea31155b49fe1c68db6918911c12.jpg)

![](images/ebf16c3d21477f1e95c6027b166678599fddda33c3258b8a50cf8681f6941c1e.jpg)

Chenda Li (Member, IEEE) received the B.Eng. degree in electronic information engineering from Huazhong University of Science and Technology, Wuhan, China, in 2018, and the M.S. and Ph.D. degrees in computer science and technology from Shanghai Jiao Tong University, Shanghai, China, in 2020 and 2024, respectively. He was a Visiting Scholar at Carnegie Mellon University from 2023 to 2024. He is currently an Assistant Professor with the AudioCC Lab, School of Computer Science, Shanghai Jiao Tong University. He has authored or

Jiahe Wang (Student Member, IEEE) received the B.Eng. degree from the Faculty of Electronic and Information Engineering, Xi’an Jiaotong University, Xi’an, China, in 2024, and is currently working toward the M.Sc. degree with the AudioCC Lab, School of Computer Science, Shanghai Jiao Tong University, Shanghai, China, under the supervision of Professor Yanmin Qian. His current research interests include speech enhancement (SE) and speech quality assessment (SQA).

coauthored more than 50 papers in peer-reviewed journals and conferences, including T-ASLP, ICASSP, INTERSPEECH, SLT, and ASRU. He is a core developer and maintainer of ESPnet-SE, an open-source speech enhancement and separation toolkit, and a co-organizer of the URGENT speech enhancement challenge series. He was selected for the 2025 Microsoft Research Asia StarTrack Program. His research interests include speech enhancement and separation, sound separation, multimodal speech and audio processing, speech language models, and generative audio models.

![](images/c1f716f90afa324dfb2555e716696232c9069d89cec7acc5cb2090aba6588f8c.jpg)

![](images/052a4add7914c16b36e1c2a7df58a34d1c126881e12e6fac09ceec62a7846cd8.jpg)

Tianhua Li (Student Member, IEEE) received the B.Eng. degree from the School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University, Shanghai, China, in 2025. He is currently working toward the M.Sc. degree with the AudioCC Lab, School of Computer Science, Shanghai Jiao Tong University, Shanghai, China, under the supervision of Yanmin Qian. His current research interests include Sound Separation, Speech Language Model, and Audio Editing.

![](images/f344e8b930647a0a6f45d5860998f0b35666fc683e252bbb74163f1551513349.jpg)

Yiming Chen is currently pursuing the Ph.D. degree with the School of Computer Science, Shanghai Jiao Tong University, Shanghai, China, where he received the B.E. degree in computer science in 2026. He conducts his research at the Auditory Cognition and Computational Acoustics (AudioCC) Lab, led by Prof. Yanmin Qian. His research interests include machine learning and its applications, particularly speech language models (SLMs), with current work focusing on multi-speaker speech recognition and speech separation.

Wei Wang received the the Ph.D. degree in computer science and technology from Shanghai Jiao Tong University, Shanghai, China, in 2026, under the supervision of Yanmin Qian. His research interests include automatic speech recognition, spokenlanguage understanding, speech synthesis, speech enhancement, and speech-quality assessment.

![](images/15d60dc21557e097a97f944f038547cf7784195c385ddbaf74d05e351a7b4c43.jpg)

Xin Zhou received the B.Eng. degree in Computer Science and Technology from the University of Electronic Science and Technology of China, Chengdu, China, in 2023, and the Master’s degree in Electronic and Information Engineering from Shanghai Jiao Tong University, Shanghai, China, in 2026, under the supervision of Yanmin Qian. He is currently an engineer at the Shanghai Artificial Intelligence Laboratory, Shanghai, China. His research interests include speech separation.

![](images/86c5cd47ae2adf3547d2ebc85fd028cc1af9bdf5e7c69bc52899a07246fb955b.jpg)

Yanmin Qian (Senior Member, IEEE) received the B.S. degree from the Department of Electronic and Information Engineering, Huazhong University of Science and Technology, Wuhan, China, in 2007, and the Ph.D. degree from the Department of Electronic Engineering, Tsinghua University, Beijing, China, in 2012. Since 2013, he has been with the School of Computer Science, Shanghai Jiao Tong University, Shanghai, China, where he is currently a Full Professor. From 2015 to 2016, he was an Associate Research with the Speech Group, Cambridge

University Engineering Department, Cambridge, U.K. He has authored or coauthored more than 300 papers in peer-reviewed journals and conferences on speech and language processing, including T-ASLP, Speech Communication, ICASSP, INTERSPEECH and ASRU. He has applied for more than 120 Chinese and American patents and won 6 championships of international challenges. His current research interests include speech and audio signal processing, automatic speech recognition and translation, speaker and language recognition, speech separation and enhancement, music generation and understanding, speech emotion perception, multimodal information processing, natural language understanding, deep learning and multi-media signal processing. He was the recipient of several top academic awards in China, including Chang Jiang Scholars Program of the Ministry of Education, Excellent Youth Fund of the National Natural Science Foundation of China, and the First Prize of Wu Wenjun Artificial Intelligence Science and Technology Award (First Completion). He was also the recipient of several awards from international research committee, including the Best Paper Awards in IEEE Transactions on Audio, Speech and Language Processing (2025), Speech Communication (2019) and Best Paper Awards from IEEE ISCSLP 2024, IEEE DCASE 2024, IEEE ASRU 2019, IEEE ISCSLP 2016. He is also the Member of IEEE Signal Processing Society Speech and Language Technical Committee.