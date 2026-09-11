# ZIPCODEC: ULTRA-LOW-FRAME-RATE STREAMING SPEECH CODING

Luca Della Libera<sup>1,2</sup>, Cem Subakan<sup>3,1,2</sup>, Mirco Ravanelli<sup>1,2</sup>

<sup>1</sup>Concordia University, <sup>2</sup>Mila-Quebec AI Institute, <sup>3</sup>Universite Laval ´

## ABSTRACT

Neural audio codecs are a fundamental component of modern speech generation systems. While recent codecs achieve increasingly low bitrates, reducing frame rate remains challenging, as each token must preserve more information while maintaining reconstruction quality. We present ZipCodec, a streaming neural speech codec operating at 6.25 Hz and 0.80 kbps with a theoretical latency of 160 ms. Our approach combines large-scale WavLM distillation with a redesigned transformer-based architecture, a scalar spherical quantizer, and a latency-aware streaming decoder. Experiments show that ZipCodec substantially outperforms existing streaming codecs at comparable bitrates in both reconstruction and downstream tasks, while operating at a significantly lower frame rate. Despite its 842M parameters, ZipCodec achieves real-time single-stream inference on a consumergrade CPU. Demo samples, code and checkpoints are available at https://lucadellalib.github.io/zipcodec-web/.

Index Terms— Speech coding, discrete tokens, streamability

## 1. INTRODUCTION

Neural audio codecs [1–3] have become a key component of modern speech generation systems, providing compact discrete representations of speech that can be modeled autoregressively. Building on the success of large language models [4–8], this discrete-token paradigm has been extended from text to speech, enabling a new generation of speech-native language models [9–13].

Recent neural codec research has explored several directions towards more compact and expressive speech representations, including lower frame rates, single-codebook designs, semantic distillation, and supervised fine-tuning [14–23]. However, simultaneously achieving a low bitrate, rich semantic and acoustic representations, high reconstruction quality, and streamability remains challenging.

Among the factors determining bitrate, frame rate is arguably the most critical for speech language modeling, as it directly determines the length of the resulting token sequence. Reducing the frame rate therefore shortens the sequence, lowering computational cost and simplifying sequence modeling. At the same time, it creates an increasingly severe information bottleneck: each discrete token must encode linguistic content, speaker characteristics, prosody, and fine acoustic details over a longer temporal interval.

Recent codecs have shown that extremely low frame rates are possible when relaxing some of these requirements. U-Codec [24], for example, operates at 5 Hz but is designed for offline acoustic reconstruction. TaDiCodec [25] reaches 6.25 Hz by leveraging text as additional side information for speech reconstruction. Flexi-Codec [26] and DyCAST [27] instead adopt variable-frame-rate representations that can reach similarly low average frame rates, but operate offline and exhibit increasing reconstruction degradation as the frame rate is reduced. Among streaming codecs that jointly capture semantic and acoustic information, Mimi [10] operates at 12.5 Hz.

To the best of our knowledge, no such codec has been demonstrated below 12.5 Hz, leaving open how far the frame rate of a streaming codec can be reduced.

In this work, we show that this limit can be pushed further by introducing ZipCodec, a streaming speech codec operating at 6.25 Hz. Building on FocalCodec-Stream [22], ZipCodec distills WavLM [28] layer-6 representations into a causal backbone, while rethinking the codec architecture and further scaling both model capacity and training data. These advances enable ZipCodec to compress speech at 0.80 kbps while substantially improving both reconstruction and representation quality. Despite its extremely low frame rate, ZipCodec remains fully streamable, with a theoretical latency of 160 ms, which is below the 200 ms timescale of typical conversational turn transitions [29], making the proposed representations compatible with highly responsive streaming speech-to-speech systems. Our main contributions are as follows:

• We introduce ZipCodec, a streaming neural speech codec that compresses speech at an exceptionally low frame rate of 6.25 Hz and a bitrate of 0.80 kbps, with a theoretical latency of 160 ms. Despite its 842M parameters, ZipCodec supports real-time inference for a single stream on a consumer-grade CPU.

• To achieve this, we substantially redesign the FocalCodec-Stream architecture for improved scalability and efficiency. In particular, we adopt an optimized transformer-based architecture that removes normalization layers and positional encodings, employ scalar spherical quantization [27] to obtain a compact factorized bottleneck, and introduce a latency-aware streaming decoder. We further scale WavLM layer-6 distillation to approximately 94,000 hours of English speech and reproduce WavLM noise and overlapping-speech augmentation strategy to better match WavLM original data distribution.

• We extensively evaluate ZipCodec across reconstruction and downstream tasks, demonstrating substantial improvements in both reconstruction and representation quality over other streaming codecs at matched bitrates, despite operating at a significantly lower frame rate.

## 2. ZIPCODEC

## 2.1. Architecture

Our streaming codec builds upon the FocalCodec-Stream [22] architecture, with several modifications designed to improve scalability and reconstruction quality. Following the same overall structure, ZipCodec consists of five main components (see Figure 1): encoder, compressor, quantizer, decompressor, and decoder.

Encoder. In contrast to FocalCodec-Stream, which employs a learned convolutional encoder followed by transformer blocks, Zip-Codec uses a causal log-mel frontend. We extract 80-dimensional log-mel features using a 25 ms Hann window and a 10 ms hop, resulting in a 100 Hz feature sequence. This simple frontend removes the need for a learned waveform encoder while providing compact representations to the subsequent compressor.

![](images/197dba5b7f014302b54ddeec588c20cf66c1d2d841d5b3926b3a1a97d81f0e65.jpg)  
Fig. 1. ZipCodec architecture. The encoder extracts features containing both acoustic and semantic information. These features are then mapped to a low-dimensional space by the compressor, quantized, and projected back by the decompressor. The decoder resynthesizes the waveform from these features. All these modules are causal, while a non-causal teacher is used for distillation to align causal features with their non-causal counterparts.

Compressor. The compressor first employs a temporal patching module that groups 16 consecutive log-mel frames and linearly projects them to the model dimension. This reduces the frame rate by a factor of 16, from 100 Hz to 6.25 Hz, with each representation spanning 160 ms of speech. The resulting sequence is then processed by a causal transformer-based backbone.

While FocalCodec-Stream relies on focal modulation [30, 31] as its main sequence modeling operator, we adopt a transformerbased design to leverage the highly optimized attention and matrixmultiplication primitives available on modern hardware. Rather than using a standard transformer, we introduce ErfFormer, which builds on LLaMA-style blocks [4] with grouped-query attention and gated feed-forward networks with SiLU activations, while making two key modifications for efficient streaming speech modeling: eliminating both normalization layers and positional encodings. Specifically, ErfFormer replaces RMSNorm [32] with DynamicErf [33], a lightweight activation that has been shown to match or outperform normalization-based architectures at lower computational cost. Erf-Former also removes positional encodings entirely. Unlike discrete text tokens, continuous acoustic representations already carry local temporal structure, while causal attention preserves their temporal ordering. More importantly, removing position-dependent representations facilitates long-running streaming inference, allowing the model to operate beyond the context lengths observed during training without requiring positional extrapolation.

Quantizer. We discretize the compressor output using scalar spherical quantization (SSQ) [27]. Given a compressor representation, we first project it to an L = 64-dimensional latent space and normalize it to the unit hypersphere. Each latent dimension is then independently quantized to one of $K = 4$ uniformly spaced scalar levels in the interval $[ - 1 / \sqrt { L } , \ 1 / \sqrt { L } ]$ . The resulting representation contains 64 two-bit symbols per frame, corresponding to a bitrate of 0.80 kbps at a frame rate of 6.25 Hz. Despite its factorized structure, SSQ implicitly defines $K ^ { L } = 4 ^ { 6 4 }$ possible joint codewords without requiring an explicit codebook of this size. The quantized representation is then renormalized to the unit hypersphere and projected back to the model dimension before being passed to the decompressor.

Decompressor. The decompressor mirrors the architecture of the compressor, employing the same ErfFormer backbone followed by a temporal unpatching module. The quantized representations are first processed by ErfFormer and then linearly projected and unpatched, expanding each 6.25 Hz representation into 8 1024- dimensional WavLM layer-6 representations at 50 Hz. Thus, each 160 ms discrete representation is mapped to a sequence of 8 continuous representations spaced 20 ms apart, which are then passed to the waveform decoder.

Decoder. The 50 Hz WavLM representations are converted to waveform samples using a streaming Vocos [34] decoder. We adapt its inverse STFT synthesis to streaming inference by maintaining the overlap-add state across consecutive decoding steps. The exceptionally low frame rate of ZipCodec also allows us to relax the causality constraints within each decoding step. Since all 8 corresponding WavLM representations are available within the same 160 ms interval, we process them jointly rather than restricting the decoder to causal processing at the 20 ms feature granularity. Specifically, we use left and right convolutional padding while constraining the overall receptive field to the 160 ms decoding window. This allows the decoder to leverage future context within each step without increasing the theoretical latency imposed by the 6.25 Hz bottleneck.

## 2.2. Training

ZipCodec follows a similar training strategy to FocalCodec-Stream, in which the encoder-compressor-quantizer-decompressor pipeline is trained by distilling WavLM features, while the waveform decoder is trained separately on continuous WavLM representations. However, we considerably simplify the distillation procedure: rather than the four-stage approach used in FocalCodec-Stream, we jointly train the encoder, compressor, quantizer, and decompressor in a single stage to reconstruct continuous WavLM layer-6 representations. We further scale both model capacity and training data, using approximately 94,000 hours of speech from LibriLight [35], VoxPopuli [36], and GigaSpeech [37], closely matching the data distribution used for WavLM pretraining.

To further reduce the distribution gap between WavLM pretraining and distillation, we reproduce its noise and overlapping-speech augmentation strategy. Each training utterance is augmented with probability 0.2 by mixing a randomly selected segment with either another utterance, simulating overlapping speech, or noise from the DNS [38] dataset. Conditioned on augmentation, DNS noise is selected with probability 0.1 and mixed at an energy ratio uniformly sampled between −5 and 20 dB; otherwise, another training utterance is mixed at a ratio between −5 and 5 dB. Importantly, the same augmented waveform is provided to both ZipCodec and the frozen WavLM teacher, such that ZipCodec learns to reconstruct the WavLM representations of the augmented speech itself.

## 3. EXPERIMENTAL SETUP

We train the ZipCodec encoder, compressor, quantizer, and decompressor on 4 NVIDIA H100 (80 GB) GPUs with a batch size of 4 per GPU, for a total batch size of 16. Training examples consist of 40.96 s segments sampled from LibriLight, VoxPopuli, and GigaSpeech proportionally to their approximate corpus sizes. Utterances longer than 40.96 s are randomly cropped, while shorter utterances are repeated and randomly cropped to the target duration; utterances shorter than 2 s are discarded. Training shards are continuously resampled and shuffled rather than traversed in fixed epochs. In addition to the main L2 reconstruction loss, we employ an entropy loss to encourage high utilization of the scalar quantization levels. We use AdamW [39] with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8$ , a peak learning rate of $2 \times 1 0 ^ { - 4 }$ , and a weight decay of 0.01. The learning rate is linearly warmed up for 10,000 optimization steps and then decayed to $2 \times 1 0 ^ { - 5 }$ following a cosine schedule. Gradients are clipped to a global norm of 1.0, and training is performed using bfloat16 mixed precision for a total of 4M optimization steps.

Table 1. Codecs considered in our experiments.
<table><tr><td>Codec</td><td>Frame Rate Bitrate Sample Rate (Hz)</td><td>(kbps)</td><td>(kHz)</td><td>Codebooks</td><td>Latency (ms)</td><td>Params (M)</td></tr><tr><td>EnCodec</td><td>75.0</td><td>1.50</td><td>24</td><td> $2 \times 1 0 2 4$ </td><td>13</td><td>15</td></tr><tr><td>AudioDec</td><td>80.0</td><td>1.60</td><td>24</td><td> $2 \times 1 0 2 4$ </td><td>13</td><td>8</td></tr><tr><td>HILCodec</td><td>75.0</td><td>1.50</td><td>24</td><td> $2 \times 1 0 2 4$ </td><td>13</td><td>11</td></tr><tr><td>Mimi</td><td>12.5</td><td>0.83</td><td>24</td><td> $6 \times 2 0 4 8$ </td><td>80</td><td>82</td></tr><tr><td>PAST</td><td>50.0</td><td>1.00</td><td>16</td><td> $2 \times 1 0 2 4$ </td><td>20</td><td>126</td></tr><tr><td>FocalCodec-S@50</td><td>50.0</td><td>0.80</td><td>16/24</td><td> $1 \times 6 5 5 3 6$ </td><td>80</td><td>249</td></tr><tr><td>ZipCodec</td><td>6.25</td><td>0.80</td><td>16</td><td> $6 4 \times 4$ </td><td>160</td><td>842</td></tr><tr><td>FocalCodec@50</td><td>50.0</td><td>0.65</td><td>16</td><td>1 × 8192</td><td>一</td><td>142</td></tr></table>

The compressor consists of 6 ErfFormer blocks with a model dimension of 2048 and a feed-forward dimension of 8192. Attention employs 16 query heads and 4 key-value heads, each with a head dimension of 128. The decompressor mirrors this architecture, using the same number of layers and dimensions. During streaming inference, both modules maintain a bounded key-value cache of 256 frames, corresponding to 40.96 s of context at 6.25 Hz and matching the context length used during training. For arbitrarily long streams, we reset the cache every 256 frames rather than using a sliding window, which would expose the model to attention patterns across longer sequences not encountered during training.

For the decoder, we build upon the original Vocos training recipe and train on LibriTTS-100 [40], resampled to 16 kHz. The decoder consists of 20 ConvNeXt blocks with a hidden dimension of 1024 and a kernel size of 7, followed by a complex spectral head and inverse STFT synthesis. All temporal components maintain explicit streaming state, including the convolutional and overlap-add states. At each streaming step, the decoder consumes 8 WavLM representations and produces 2,560 waveform samples, corresponding to 160 ms of audio. We employ the multi-scale and multi-period discriminators from [41], together with the multi-resolution discriminator from [3]. We additionally incorporate the speaker consistency loss proposed in [16], using WavLM-base-SV [28] as a pretrained speaker embedding extractor. We train on 7,040-sample audio segments with a batch size of 16 using AdamW with $\beta _ { 1 } = 0 . 8$ $\beta _ { 2 } = 0 . 9 9$ , an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , and a weight decay of 0.01. The learning rate follows an exponential decay schedule with a factor of 0.999. Training continues until perceptual quality saturates, which occurs after approximately 5M optimization steps.

## 4. RESULTS

We adopt the evaluation protocol of FocalCodec-Stream [22], focusing on streaming codecs operating in the low-bitrate regime. For models supporting multiple quantizer configurations, we select the setting closest to ZipCodec bitrate of 0.80 kbps to enable comparisons under similar compression constraints. Our baselines include the acoustic codecs EnCodec [2], AudioDec [42], and HILCodec [43], as well as the hybrid codecs Mimi [10] and PAST [44], which incorporate semantic information through distillation and supervised fine-tuning, respectively. We further compare against FocalCodec-Stream, the closest baseline to ZipCodec in terms of system design. Finally, we include the original nonstreaming FocalCodec@50 [20] as an offline reference for WavLM layer-6 distillation. The configurations of all evaluated codecs are summarized in Table 1.

## 4.1. Speech Resynthesis and Voice Conversion

We first evaluate ZipCodec on speech resynthesis (SR) in both English and multilingual settings, using the evaluation protocol introduced in [20]. We evaluate English resynthesis on LibriSpeech [45] test-clean and multilingual resynthesis on a subset of MLS [46]. Reconstruction quality is measured along several dimensions. UTMOS [47] evaluates perceptual naturalness, while intelligibility is assessed through dWER, computed as the WER between Whisper-small [48] transcriptions of the original and reconstructed speech. Speaker preservation is quantified using WavLM-based embedding similarity (Sim). We additionally report code usage and normalized entropy to quantify codebook utilization, as well as the real-time factor (RTF) to measure inference efficiency. RTF is measured on a 1/8 partition of an NVIDIA H100 (80 GB) GPU using multi-instance GPU partitioning.

As shown in Table 2, ZipCodec achieves the strongest overall speech resynthesis performance among streaming codecs in both the English and multilingual settings, consistently improving over FocalCodec-Stream in perceptual quality, intelligibility, and speaker fidelity. The improvements are particularly pronounced in the multilingual setting, despite ZipCodec operating at only 6.25 Hz compared to 50 Hz for FocalCodec-Stream and at least 12.5 Hz for the other streaming baselines. ZipCodec also achieves full code utilization in both settings while maintaining high normalized entropy, indicating effective use of the factorized SSQ bottleneck. Finally, despite its substantially lower frame rate, ZipCodec approaches the reconstruction quality of the non-streaming FocalCodec@50 reference, narrowing the gap in perceptual quality and intelligibility while surpassing it in speaker fidelity.

We also perform one-shot voice conversion (VC) experiments to assess the ability of ZipCodec to disentangle linguistic content from speaker information. Following the protocol of [22], we use a dataset of parallel utterances derived from VCTK [49]. As shown in Table 2, ZipCodec achieves the highest perceptual quality among streaming codecs while maintaining strong speaker fidelity, second only to FocalCodec-Stream. It also preserves competitive intelligibility, outperforming most streaming baselines, with only FocalCodec-Stream and PAST achieving lower dWER. Overall, these results show that the 6.25 Hz bottleneck retains sufficient information for effective voice conversion despite its aggressive temporal compression.

## 4.2. Downstream Tasks

To evaluate the quality of the learned representations beyond reconstruction, we consider the same downstream tasks and experimental protocol as FocalCodec-Stream [22], following the DASB benchmark [50]. We evaluate five discriminative tasks: automatic speech recognition (ASR) and speaker identification (SI) on LibriSpeech-460, speech emotion recognition (SER) on IEMOCAP [51], keyword spotting (KS) on Speech Commands [52], and intent classification (IC) on SLURP [53]. We additionally consider two generative tasks: speech enhancement (SE) on VoiceBank [54] and speech separation (SS) on Libri2Mix-100 [55]. We use the same shallow LSTM-based probes for discriminative tasks and non-autoregressive Conformer models for the generative tasks. For discriminative tasks, we use the representations reconstructed after the quantization bottleneck and before the decoder. In ZipCodec, these correspond to the 50 Hz representations produced by the decompressor. For generative tasks, we instead use the ZipCodec representations before temporal unpatching, allowing the SE and SS models to operate directly at the native 6.25 Hz frame rate and benefit from the short sequence length provided by its low-rate bottleneck. We report error rates for the discriminative tasks and DNSMOS [56], dWER, and speaker similarity for SE and SS. We refer to [20, 22] for further details on the downstream evaluation setup.

Table 2. Speech resynthesis and voice conversion. Best, second-best and best non-streaming results are highlighted.
<table><tr><td rowspan="2">Codec</td><td rowspan="2" colspan="2">Frame Bitrate Rate (Hz) (kbps)</td><td colspan="6">SR - English Code</td><td colspan="6">SR – Multilingual</td><td colspan="3">VC</td></tr><tr><td colspan="3">UTMOS ↑ dWER ↓ Sim ↑</td><td colspan="3">Norm. 个 Entropy</td><td colspan="3">UTMOS ↑ dWER ↓ Sim ↑</td><td colspan="3"> $\begin{array} { c } { \mathbf { C o d e } } \\ { \mathbf { U s a g e } } \end{array}$  Norm. ← Entropy</td><td colspan="3">UTMOS ↑ dWER ↓ Sim ↑</td></tr><tr><td>Reference</td><td></td><td></td><td>4.09</td><td>0.00</td><td>100.0</td><td> $\mathbf { U s a g e }$ </td><td></td><td></td><td>2.84</td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td>4.09</td><td>0.00</td><td>100.0</td></tr><tr><td>EnCodec</td><td>75.0</td><td>1.50</td><td>1.58</td><td>8.08</td><td>93.8</td><td>93.4</td><td>82.1</td><td>91</td><td>1.33</td><td>29.60</td><td>95.5</td><td>93.4</td><td>79.2</td><td>113</td><td>1.24</td><td>86.52</td><td>72.2</td></tr><tr><td>AudioDec</td><td>80.0</td><td>1.60</td><td>1.48</td><td>11.61</td><td>92.1</td><td>91.9</td><td>70.0</td><td>145</td><td>1.29</td><td>40.95</td><td>92.3</td><td>87.5</td><td>68.2</td><td>195</td><td>1.26</td><td>68.45</td><td>68.2</td></tr><tr><td>HILCodec</td><td>75.0</td><td>1.50</td><td>2.86</td><td>6.65</td><td>95.4</td><td>99.0</td><td>95.6</td><td>41</td><td>1.81</td><td>25.32</td><td>97.8</td><td>99.1</td><td>94.8</td><td>41</td><td>1.40</td><td>58.36</td><td>76.8</td></tr><tr><td>Mimi</td><td>12.5</td><td>0.83</td><td>3.44</td><td>4.77</td><td>96.6</td><td>96.2</td><td>92.0</td><td>154</td><td>2.19</td><td>26.12</td><td>97.4</td><td>96.5</td><td>89.2</td><td>216</td><td>2.62</td><td>110.00</td><td>91.3</td></tr><tr><td>PAST</td><td>50.0</td><td>1.00</td><td>2.33</td><td>4.04</td><td>83.8</td><td>56.7</td><td>90.7</td><td>59</td><td>1.44</td><td>49.35</td><td>80.8</td><td>57.0</td><td>87.5</td><td>63</td><td>1.42</td><td>18.28</td><td>68.5</td></tr><tr><td>FocalCodec-S@50</td><td>50.0</td><td>0.80</td><td>3.85</td><td>3.68</td><td>97.0</td><td>100.0</td><td>98.7</td><td>106</td><td>2.65</td><td>19.88</td><td>98.1</td><td>99.2</td><td>98.3</td><td>107</td><td>3.10</td><td>22.71</td><td>92.5</td></tr><tr><td>ZipCodec</td><td>6.25</td><td>0.80</td><td>3.89</td><td>2.83</td><td>98.0</td><td>100.0</td><td>96.2</td><td>62</td><td>2.69</td><td>15.52</td><td>98.7</td><td>100.0</td><td>96.5</td><td>106</td><td>3.15</td><td>25.90</td><td>91.5</td></tr><tr><td>FocalCodec@50</td><td>50.0</td><td>0.65</td><td>4.05</td><td>2.18</td><td>97.4</td><td>100.0</td><td>98.9</td><td>123</td><td>2.96</td><td>12.57</td><td>98.3</td><td>100.0</td><td>98.1</td><td>116</td><td>3.38</td><td>21.27</td><td>92.2</td></tr></table>

Table 3. Discriminative and generative downstream tasks. Best, second-best and best non-streaming results are highlighted.
<table><tr><td rowspan="2">Codec</td><td rowspan="2">Frame Rate (Hz)</td><td rowspan="2">Bitrate (kbps)</td><td>ASR</td><td>SI</td><td>SER</td><td>KS</td><td>IC</td><td colspan="4">SE</td><td colspan="3">SS</td></tr><tr><td>WER↓</td><td>ER↓</td><td>ER↓</td><td>ER↓</td><td>ER↓</td><td></td><td>DNSMOS ↑</td><td>dWER↓</td><td>Sim ↑</td><td>DNSMOS ↑</td><td>dWER↓</td><td>Sim ↑</td></tr><tr><td>Reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td><td></td><td>100.0</td><td>3.77</td><td>0.00</td><td>100.0</td></tr><tr><td>EnCodec</td><td>75.0</td><td>1.50</td><td>28.55</td><td>3.25</td><td>41.94</td><td>96.16</td><td></td><td>49.79</td><td>3.56 3.13</td><td>37.31</td><td>85.6</td><td>3.11</td><td>77.61</td><td>87.4</td></tr><tr><td>AudioDec</td><td>80.0</td><td>1.60</td><td>29.21</td><td>1.69</td><td>45.85</td><td>25.30</td><td></td><td>46.77</td><td>2.96</td><td>61.11</td><td>84.3</td><td>2.97</td><td>88.59</td><td>84.0</td></tr><tr><td>HILCodec</td><td>75.0</td><td>1.50</td><td>29.89</td><td>1.98</td><td>51.61</td><td>15.17</td><td></td><td>53.69</td><td>3.32</td><td>41.33</td><td>90.2</td><td>3.35</td><td>78.43</td><td>86.9</td></tr><tr><td>Mimi</td><td>12.5</td><td>0.83</td><td>22.56</td><td>3.13</td><td>35.71</td><td>5.81</td><td></td><td>35.74</td><td>3.14</td><td>55.99</td><td>86.7</td><td>3.32</td><td>86.49</td><td>88.9</td></tr><tr><td>PAST FocalCodec-S@50</td><td>50.0 50.0</td><td>1.00 0.80</td><td>10.74 17.02</td><td>3.43 2.18</td><td>36.41 34.56</td><td>6.41</td><td></td><td>31.66 29.49</td><td>3.15 3.56</td><td>18.19 19.56</td><td>77.9 87.7</td><td>3.15 3.68</td><td>85.61 75.43</td><td>80.3 90.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>5.63</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ZipCodec</td><td>6.25†</td><td>0.80</td><td>16.07</td><td>0.49</td><td>33.64</td><td>3.95</td><td>26.76</td><td></td><td>3.60</td><td>20.10</td><td>91.0</td><td>3.77</td><td>66.67</td><td>91.0</td></tr><tr><td>FocalCodec@50</td><td>50.0</td><td>0.65</td><td>15.33</td><td>0.35</td><td>34.79</td><td>4.23</td><td>24.66</td><td></td><td>3.52</td><td>12.35</td><td>90.4</td><td>3.71</td><td>72.61</td><td>89.5</td></tr></table>

<sup>†</sup>Discriminative tasks use the 50 Hz representations after temporal unpatching, while generative tasks use the 6.25 Hz representations before temporal unpatching.

Table 4. Streaming efficiency of ZipCodec.
<table><tr><td>Device</td><td>Batch Size</td><td>RTF↑</td><td>Latency p99 (ms) ↓</td><td>VRAM (GiB) ↓</td></tr><tr><td>CPU</td><td>1</td><td>1.33</td><td>124.68</td><td>一</td></tr><tr><td>GPU</td><td>1</td><td>13.96</td><td>12.71</td><td>3.33</td></tr><tr><td>GPU</td><td>2</td><td>11.43</td><td>15.44</td><td>3.35</td></tr><tr><td>GPU</td><td>4</td><td>9.29</td><td>18.86</td><td>3.40</td></tr><tr><td>GPU</td><td>8</td><td>5.85</td><td>28.04</td><td>3.56</td></tr><tr><td>GPU</td><td>16</td><td>4.49</td><td>43.41</td><td>3.84</td></tr></table>

Results are reported in Table 3. On discriminative tasks, Zip-Codec achieves the best performance among streaming codecs on SI, SER, KS, and IC, while obtaining the second-best ASR result. It consistently improves over FocalCodec-Stream across all five tasks, with particularly large gains in SI and KS. ZipCodec also matches or surpasses the non-streaming FocalCodec@50 reference on SER and KS, while remaining competitive on the other tasks. These results show that the representations reconstructed from the 6.25 Hz bottleneck retain rich linguistic, speaker, and paralinguistic information.

The generative results further demonstrate the effectiveness of the low-frame-rate representation. ZipCodec achieves the strongest overall performance among streaming codecs on both tasks. For SE, it achieves the highest perceptual quality and speaker similarity while maintaining competitive intelligibility. For SS, it consistently outperforms both FocalCodec-Stream and the non-streaming baseline. These results are particularly promising for downstream generative modeling, as ZipCodec combines strong representation quality with sequences that are 8 times shorter than the 50 Hz representations used by FocalCodec-Stream.

## 4.3. Streaming Efficiency

We evaluate the streaming efficiency of ZipCodec on a machine equipped with an Intel i7-10875H CPU with 8 cores @ 2.30 GHz, 32 GB of RAM, and an NVIDIA GeForce RTX 3070 (8 GB)

GPU. Measurements are performed on 40.96 s sequences, corresponding to the maximum context maintained by ZipCodec before resetting the key-value cache. We report RTF, together with the 99th-percentile latency of each 160 ms streaming step and peak GPU memory consumption. We evaluate batch size 1 on both CPU and GPU and additionally vary the GPU batch size to assess efficiency under concurrent streams. The reported metrics exclude system-level overheads such as audio capture and playback buffering, host-device transfers, resampling, and network transport.

As shown in Table 4, despite its 842M parameters, ZipCodec supports real-time single-stream inference on a consumer-grade CPU, achieving an RTF of 1.33 with a p99 latency below the 160 ms codec frame duration. GPU inference is substantially faster and scales efficiently to concurrent streams, maintaining real-time performance even at a batch size of 16 while requiring only a modest increase in memory. These results show that the low frame rate of ZipCodec enables efficient streaming despite its large model size.

## 5. CONCLUSION

We introduced ZipCodec, a streaming neural speech codec operating at 6.25 Hz and 0.80 kbps with a theoretical latency of 160 ms. By combining large-scale WavLM distillation with a new architecture designed for efficient streaming, ZipCodec achieves strong reconstruction and representation quality while substantially reducing the frame rate compared with existing streaming codecs. Experiments show consistent improvements over streaming baselines across reconstruction and downstream tasks, while enabling real-time inference on a consumer-grade CPU.

## 6. ACKNOWLEDGMENTS

We gratefully acknowledge the support of NSERC, the Digital Research Alliance of Canada (alliancecan.ca), Translated (Imminent Program), and Apple (Seed Grant) through research funding, computing resources, and donations.

## 7. REFERENCES

[1] N. Zeghidour, A. Luebs, et al., “SoundStream: An end-to-end neural audio codec,” IEEE/ACM TASLP, pp. 495–507, 2021.

[2] A. Defossez, J. Copet, et al., “High fidelity neural audio compression,”´ TMLR, 2023.

[3] R. Kumar, P. Seetharaman, et al., “High-fidelity audio compression with improved RVQGAN,” in NeurIPS, 2023.

[4] A. Grattafiori, A. Dubey, et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[5] A. Q. Jiang, A. Sablayrolles, et al., “Mixtral of experts,” arXiv preprint arXiv:2401.04088, 2024.

[6] G. Comanici, E. Bieber, et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[7] A. Singh, A. Fry, et al., “OpenAI GPT-5 system card,” arXiv preprint arXiv:2601.03267, 2025.

[8] DeepSeek-AI, A. Liu, et al., “DeepSeek-V3 technical report,” arXiv preprint arXiv:2412.19437, 2025.

[9] M. Hassid, T. Remez, et al., “Textually pretrained speech language models,” in ICLR, 2023.

[10] A. Defossez, L. Mazar ´ e, et al., “Moshi: A speech-text foundation´ model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[11] T. A. Nguyen, B. Muller, et al., “SpiRit-LM: Interleaved spoken and written language model,” TACL, vol. 13, pp. 30–52, 2025.

[12] S. J. Park, J. Salazar, et al., “Long-form speech generation with spoken language models,” in ICML, 2025.

[13] L. Della Libera, C. Subakan, et al., “WavSLM: Single-stream speech language modeling via WavLM distillation,” in Interspeech, 2026.

[14] X. Zhang, D. Zhang, et al., “SpeechTokenizer: Unified speech tokenizer for speech large language models,” in ICLR, 2024.

[15] J. D. Parker, A. Smirnov, et al., “Scaling transformers for low-bitrate high-quality speech coding,” in ICLR, 2025.

[16] E. Casanova, P. Neekhara, et al., “NanoCodec: Towards high-quality ultra fast speech LLM inference,” in Interspeech, 2025, pp. 5028–5032.

[17] S. Ji, Z. Jiang, et al., “WavTokenizer: An efficient acoustic discrete codec tokenizer for audio language modeling,” in ICLR, 2025.

[18] D. Xin, X. Tan, et al., “BigCodec: Pushing the limits of low-bitrate neural speech codec,” arXiv preprint arXiv:2409.05377, 2024.

[19] H. Wu, N. Kanda, et al., “TS3-Codec: Transformer-based simple streaming single codec,” in Interspeech, 2025, pp. 604–608.

[20] L. Della Libera, F. Paissan, et al., “FocalCodec: Low-bitrate speech coding via focal modulation networks,” in NeurIPS, 2025.

[21] Z. Ye, X. Zhu, et al., “Llasa: Scaling train-time and inferencetime compute for Llama-based speech synthesis,” arXiv preprint arXiv:2502.04128, 2025.

[22] L. Della Libera, C. Subakan, et al., “FocalCodec-Stream: Streaming low-bitrate speech coding via causal distillation,” in ICASSP, 2026, pp. 17002–17006.

[23] F. Paissan, L. Della Libera, et al., “Exploring token-space manipulation in latent audio tokenizers,” arXiv preprint arXiv:2605.11192, 2026.

[24] X. Yang, L. Zhou, et al., “U-Codec: Ultra low frame-rate neural speech codec for fast high-fidelity speech generation,” arXiv preprint arXiv:2510.16718, 2025.

[25] Y. Wang, D. Chen, et al., “TaDiCodec: Text-aware diffusion speech tokenizer for speech language modeling,” in NeurIPS, 2025.

[26] J. Li, Y. Qian, et al., “FlexiCodec: A dynamic neural audio codec for low frame rates,” in ICLR, 2025.

[27] L. Della Libera, C. Subakan, et al., “Beyond fixed frames: Dynamic character-aligned speech tokenization,” arXiv preprint arXiv:2601.23174, 2026.

[28] S. Chen, C. Wang, et al., “WavLM: Large-scale self-supervised pretraining for full stack speech processing,” IEEE JSTSP, pp. 1505–1518, 2022.

[29] S. C. Levinson and F. Torreira, “Timing in turn-taking and its implications for processing models of language,” Front. Psychol., 2015.

[30] J. Yang, C. Li, et al., “Focal modulation networks,” in NeurIPS, 2022.

[31] L. Della Libera, C. Subakan, et al., “Focal modulation networks for interpretable sound classification,” in ICASSPW, 2024, pp. 853–857.

[32] B. Zhang and R. Sennrich, “Root mean square layer normalization,” arXiv preprint arXiv:1910.07467, 2019.

[33] M. Chen, T. Lu, et al., “Stronger normalization-free transformers,” in CVPR, 2026.

[34] H. Siuzdak, “Vocos: Closing the gap between time-domain and fourierbased neural vocoders for high-quality audio synthesis,” in ICLR, 2024.

[35] J. Kahn, M. Riviere, et al., “Libri-Light: A benchmark for ASR with limited or no supervision,” in ICASSP, 2020, pp. 7669–7673.

[36] C. Wang, M. Riviere, et al., “VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in ACL, 2021, pp. 993–1003.

[37] G. Chen, S. Chai, et al., “GigaSpeech: An evolving, multi-domain ASR corpus with 10,000 hours of transcribed audio,” in Interspeech, 2021.

[38] C. K. Reddy, H. Dubey, et al., “Interspeech 2021 Deep Noise Suppression Challenge,” in Interspeech, 2021, pp. 2796–2800.

[39] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.

[40] H. Zen, V. Dang, et al., “LibriTTS: A corpus derived from LibriSpeech for text-to-speech,” in Interspeech, 2019.

[41] J. Kong, J. Kim, et al., “HiFi-GAN: generative adversarial networks for efficient and high fidelity speech synthesis,” in NeurIPS, 2020.

[42] Y.-C. Wu, I. D. Gebru, et al., “AudioDec: An open-source streaming high-fidelity neural audio codec,” in ICASSP, 2023, pp. 1–5.

[43] S. Ahn, B. J. Woo, et al., “HILCodec: High-fidelity and lightweight neural audio codec,” IEEE JSTSP, vol. 18, pp. 1517–1530, 2024.

[44] N. Har-Tuv, O. Tal, et al., “PAST: Phonetic-acoustic speech tokenizer,” in Interspeech, 2025, pp. 3509–3513.

[45] V. Panayotov, G. Chen, et al., “LibriSpeech: An ASR corpus based on public domain audio books,” in ICASSP, 2015, pp. 5206–5210.

[46] V. Pratap, Q. Xu, et al., “MLS: A large-scale multilingual dataset for speech research,” in Interspeech, 2020, pp. 2757–2761.

[47] T. Saeki, D. Xin, et al., “UTMOS: UTokyo-SaruLab system for Voice-MOS challenge 2022,” in Interspeech, 2022, pp. 4521–4525.

[48] A. Radford, J. W. Kim, et al., “Robust speech recognition via largescale weak supervision,” in ICML, 2023, vol. 202, pp. 28492–28518.

[49] J. Yamagishi, C. Veaux, et al., “CSTR VCTK corpus: English multispeaker corpus for CSTR voice cloning toolkit,” University of Edinburgh, CSTR, vol. 6, pp. 15, 2017.

[50] P. Mousavi, J. Duret, et al., “DASB - discrete audio and speech benchmark,” TMLR, 2026.

[51] C. Busso, M. Bulut, et al., “IEMOCAP: Interactive emotional dyadic motion capture database,” LREC, vol. 42, no. 4, pp. 335–359, 2008.

[52] P. Warden, “Speech Commands: A dataset for limited-vocabulary speech recognition,” arXiv preprint arXiv:1804.03209, 2018.

[53] E. Bastianelli, A. Vanzo, et al., “SLURP: A spoken language understanding resource package,” in EMNLP, 2020, pp. 7252–7262.

[54] C. Valentini-Botinhao, X. Wang, et al., “Investigating RNN-based speech enhancement methods for noise-robust text-to-speech,” in SSW, 2016, pp. 146–152.

[55] J. Cosentino, M. Pariente, et al., “LibriMix: An open-source dataset for generalizable speech separation,” arXiv preprint arXiv:2005.11262, 2020.

[56] C. K. Reddy, V. Gopal, et al., “DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,” in ICASSP, 2022.