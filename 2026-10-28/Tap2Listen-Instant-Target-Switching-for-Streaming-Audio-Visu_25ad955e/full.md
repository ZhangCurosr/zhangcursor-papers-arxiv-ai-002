# Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction

Hansol Park, Minkyung Song, Zinu Kim, Yujun Lee, Hoseong Ahn, Hyunjin Choi, Kyuhong Shim Sungkyunkwan University firstri@skku.edu, khshim@skku.edu

## Abstract

Streaming audio-visual target speaker extraction (AV-TSE) uses a target speaker’s face to extract the speaker’s speech from an input audio mixture in real time. However, existing systems fix the target speaker for an entire clip, while conversation demands frequent and immediate target switches. To address this problem, we propose Tap2Listen, a streaming AV-TSE model designed for instant response to target switches. The model separates computation by modality: the target-agnostic audio encoder processes only the mixture, while the target-specific visual cues of the selected face condition only the decoder. A switch requires only a 0.12 ms re-projection and takes effect immediately. We also design component-wise losses: a prediction loss for the encoder and a supervised extraction loss for the decoder. For evaluation, we measure how quickly output quality recovers after a switch. On the standard LRS3-2mix test set, Tap2Listen improves SI-SNRi from 8.85 to 9.68 dB at a comparable parameter count. After a switch to a known face, the model produces full-quality output from the first frame, whereas the baseline needs about 1 second to recover. Finally, we show that the proposed streaming AV-TSE model runs in real time on a single CPU core.

## 1 Introduction

As hearables and smart glasses gain cameras and on-device computation, users can directly choose whom to listen to [24–26]. Audio-visual target speaker extraction (AV-TSE) uses face information to extract a single speaker’s speech from a mixture [13, 14]. Recent causal streaming AV-TSE systems achieve strong extraction quality under real-time playback constraints [3, 15, 27]. However, these systems are typically designed for a fixed target speaker [3, 19, 27]. In practice, users in a conversation switch targets repeatedly and expect each switch to take effect immediately. For example, tapping a face on the screen should leave only the speaker audible, and tapping a different face should switch the output to the new speaker with minimal delay (Fig. 1). Nevertheless, interactive target switching has received little attention.

A key difficulty comes from two aspects of existing AV-TSE systems. First, audio and visual information are often tightly coupled throughout processing, making a change of visual target entangled with ongoing audio computation. Second, streaming systems may accumulate targetdependent state over time. After a target change, information associated with the previous speaker can persist in recurrent, autoregressive, or speaker-tracking states, delaying the transition to the new target [8, 9, 15]. Together, these properties make instant target switching difficult and motivate an architecture that separates causal and persistent audio processing from target-specific conditioning.

We propose Tap2Listen, a streaming AV-TSE model designed for instant target switching. The key idea is to separate target-agnostic and target-specific computation. A causal audio encoder processes only the audio mixture and maintains a shared temporal cache, while visual cues from the selected face condition only the decoder (Fig. 2). Switching targets does not invalidate the encoder cache and requires only re-projecting the cached memory with the newly selected face. As a result, a “tap” recomputes only the cached-window projection, immediately producing the new speaker’s output.

![](images/d14cf68bae976576832724baa84d131cc1023c2c3ef0a43556e4165296b69048.jpg)  
Figure 1: Tap2Listen interaction. Two speakers are present in the same mixture; tapping face A or B immediately switches the output to the selected speaker.

We evaluate both conventional extraction quality and the switching transition itself, using response delay and post-switch recovery curves. On LRS3-2mix [1], Tap2Listen achieves 9.68 dB SI-SNRi compared with 8.85 dB of the baseline [15] at a comparable parameter count. More importantly, the model reaches full quality from the first frame after switching to a cached face, while the baseline requires about one second to recover.

Our contributions are summarized as follows.

• We formulate interactive target switching as a distinct requirement of streaming AV-TSE and evaluate the transition using response delay and post-switch recovery, rather than relying only on utterance-level averages.

• We propose Tap2Listen, which keeps persistent audio processing target-agnostic and decouples target-specific computation from the streaming audio state, together with componentwise extraction and prediction losses for training the decomposed architecture.

• We achieve competitive extraction quality, immediate switching to cached targets without a recovery period, and real-time execution on a single mobile CPU core with Tap2Listen.

## 2 Tap2Listen: Instant Target Switching for Streaming AV-TSE

## 2.1 Architecture

Tap2Listen consists of a target-agnostic causal audio encoder that processes only the mixture and a target-conditioned decoder that uses the selected face, with 8.06 M trainable parameters (Fig. 2). The encoder projects 16 ms-window, 8 ms-hop STFT frames to 256 dimensions, followed by 6 pre-RMSNorm [28] blocks with local causal self-attention [23] over 128 frames. Because the encoder is target-agnostic, the encoder cache remains valid across target switches. Frozen visual frontends provide lip features from causal BlazeNet64 at 25 Hz [15] and face ID vectors from a face embedder [2]. A rolling buffer stores each face’s lip features over the same 4 s window.

The decoder is the only target-dependent component. For a selected face s, temporally aligned lip features are combined with the cached audio representation to form the attention memory,

$$
k _ { \tau } , v _ { \tau } = W _ { K , V } \big [ \bar { h } _ { \tau } ; \operatorname* { l i p } _ { s , \tau } \big ] , \qquad \tau \in [ t - M , t ]\tag{1}
$$

where $M = 5 0 0$ frames (4 s at the 8 ms hop), making a tap $1 . 2 \times 1 0 ^ { 8 }$ MACs. The face identity embedding contributes the decoder query:

$$
z _ { t } ^ { ( 0 ) } = W _ { X } x _ { t } + W _ { H } \bar { h } _ { t } + W _ { C } \mathrm { i d } _ { s }\tag{2}
$$

The time-varying, audio-synchronized lip features [20] provide visual cues for cross-attention to guide the temporal correspondence between lip motion and acoustics, whereas the fixed identity embedding appears only in the decoder query.

![](images/f15c8bff1c48012129943dc1edea9cf5281f72a08b2d2eba30f48530593ae229.jpg)  
Figure 2: Tap2Listen architecture. Blue denotes target-agnostic computation that is retained across target switches. Orange indicates target-specific computation recomputed at each switch. Gray blocks are frozen pretrained networks.

The query passes through K = 4 cross-attention and FFN blocks with LayerScale [22]. A shared mask head produces a complex ratio mask at each block, from which the iSTFT with overlap-add reconstructs the waveform at every 8 ms hop. The decoder uses neither self-attention over previous decoder outputs nor autoregressive feedback; causality is preserved because its memory contains only frames up to the current time.

## 2.2 Training Strategy and Loss

During training, each speaker in the mixture is treated in turn as the target. The encoder runs once for each mixture, while the memory projection is recomputed for each speaker. The shared mask head outputs a mask at each decoder block $k ,$ from which we obtain the target estimate $\hat { s } ^ { ( k ) }$ under deep supervision [7]. The extraction loss is:

$$
\mathcal { L } _ { \mathrm { e x t } } = \sum _ { s } \sum _ { k } w _ { k } \cdot \mathrm { c l a m p } \big ( - \mathrm { S I } { \cdot } \mathrm { S D R } ( \hat { s } _ { s } ^ { ( k ) } ) , - 3 0 \mathrm { d } \mathrm { B } \big )\tag{3}
$$

where $w = ( 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 )$ assigns larger weights to later decoder blocks. Each term is capped at −30 dB [18], so examples already above 30 dB SI-SDR contribute no gradient. To encourage the target-agnostic encoder to capture predictive temporal structure, we attach a linear auxiliary head on the top-layer encoder state to predict the change in log magnitude, $\Delta \log | X |$ , five frames ahead:

$$
{ \mathcal { L } } _ { \mathrm { p r e d } } = \sum _ { t } \left\| g ( { \bar { h } } _ { t } ) - \left( \log | X | _ { t + 5 } - \log | X | _ { t } \right) \right\| _ { 2 } ^ { 2 }\tag{4}
$$

This objective uses only the mixture signal and requires no target-speaker labels. The final objective is a weighted sum of two losses: $\mathcal { L } = \bar { \mathcal { L } } _ { \mathrm { e x t } } + \lambda \mathcal { L } _ { \mathrm { p r e d } }$ , with $\lambda = 0 . 1$

Table 1: Standard LRS3-2mix test (3,000 fixed mixtures). SI-SNRi and SNRi are reported in dB.
<table><tr><td rowspan="2">Model</td><td colspan="3">Setup</td><td colspan="4">Improvement</td></tr><tr><td>Params ↓</td><td>Training budget ↓</td><td>Alg. lat. ↓ |</td><td>SI-SNRi ↑</td><td>SNRi↑</td><td>PESQi ↑</td><td>STOIi↑</td></tr><tr><td>AV-SkiM [15]</td><td>8.1 M</td><td>4×V100, 150 ep</td><td>~1ms</td><td>8.85</td><td>9.16</td><td>0.52</td><td>0.19</td></tr><tr><td>Tap2Listen (ours)</td><td>8.1 M</td><td>1×3080, 150 ep</td><td>16 ms</td><td>9.68</td><td>10.05</td><td>0.71</td><td>0.20</td></tr></table>

## 2.3 Inference and Target Switching

At inference, the encoder state is retained in a shared cache. A switch from s to $s ^ { \prime }$ requires only recomputing Eq. (1) over the cached frames with the buffered lip features of $s ^ { \prime }$ , and replacing the identity term in Eq. (2), without re-running the audio encoder. Because there is no target-specific cache or state, returning to a previously selected face costs exactly the same target-specific recomputation as any other switch. In contrast, streaming systems that rely on autoregressive feedback or speakerembedding momentum inherit state from the previous target after a switch and must wait for it to decay [8, 9, 15].

## 3 Experimental Results

## 3.1 Setup

Training and Evaluation. We trained the model for 150 epochs (28 GPU-h) on a single NVIDIA RTX 3080. Training uses bf16 precision, a batch size of 16, AdamW [11] with a cosine schedule from $1 0 ^ { - 3 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 }$ . We use an exponential moving average (EMA) of the model parameters with a decay of 0.999 and report all results using the EMA weights. For comparison, we use the released AV-SkiM [15] checkpoint and re-evaluate it in our environment using the same data and evaluation pipeline as Tap2Listen. We report SI-SNRi, SNRi, PESQi, and STOIi for extraction quality, and measure selection-to-audio delay and post-switch SI-SDR recovery for target switching.

Dataset. We use the ClearerVoice LRS3-2mix protocol with 40k/5k/3k train/val/test rows from 66,974 LRS3 [1] utterances, with only the test split being speaker-independent. Nothing is pre-mixed. The target is anchored at 0 dB, and the power-matched interferer is mixed at ±10 dB SNR. In training, the interferer and SNR are redrawn from the 58.5k-utterance pool, so 40k denotes anchors, not mixtures. Switching uses our reconstruction of the AV-SkiM attention-shift protocol, splitting the same 3,000 test rows 1,014/1,012/974 no-switch/switch-face/new-face. The released checkpoint reproduces no-switch and switch-face means within 0.10 and 0.56 dB, respectively; since the original new-face seed is unavailable, all models are compared on the same reconstructed mixtures.

## 3.2 Fixed-Target Extraction

Tap2Listen achieves 9.68 dB SI-SNRi, 0.71 PESQi [17], and 0.20 STOIi [21], exceeding the baseline’s numbers under identical conditions (Table 1). This corresponds to a +0.83 dB improvement at matched parameter count while being trained on a single RTX 3080 rather than four V100 GPUs. These results show that Tap2Listen retains strong fixed-target extraction performance despite being designed primarily for interactive target switching. Tap2Listen has a 16 ms algorithmic latency, compared with approximately 1 ms for AV-SkiM. This difference mainly reflects the processing granularity: Tap2Listen operates in the STFT domain, whereas AV-SkiM operates directly on the waveform. While the latter provides finer algorithmic granularity, it requires substantially more frequent model invocations; Tap2Listen instead processes one frame every 8 ms, reducing the frequency of model invocations for practical on-device execution.

## 3.3 Target Switching

Without a target change, the two systems perform comparably: AV-SkiM reaches 9.56 dB and Tap2Listen 9.51 dB, a 0.05 dB gap well within the between-utterance variation in Fig. 3(c). After switching to a known face, Tap2Listen leads by 2.59 dB already in the first 200 ms window with no recovery period, whereas AV-SkiM needs about 1 s to close the gap (Fig. 3). For the first-appearing face, the gap reaches 8.08 dB over the first 1 s after the tap (Table 2), and peaks at 11.9 dB at 80 ms; the baseline still trails by 3.0 dB at 1.6 s. In contrast, the corresponding full-utterance gains are only 1.03 dB for a known face and 3.43 dB for a first-appearing face, indicating that utterance-level averages substantially obscure the transition gap.

![](images/818b45d1a8944144ffab74a5a590438b44b8bec249c8153cef31aa75f0a46dd4.jpg)  
(a)

![](images/d992febba1dda1286a5999c73a946a3ffbe670d5ee1185e006945f606fbe8323.jpg)  
(b)

![](images/7b0098e6a0a94e3a09d492cd749a7c4a591e3bec738cf4772de95ede775c2e84.jpg)  
(c)  
Figure 3: SI-SDR after switching, measured in 200 ms windows (mean ± s.e.m.). (a) known-face, (b) new face, and (c) no-switch control with matched pseudo-switch offsets. Shading marks the first 1 s after the switch, and dashed lines show the mean over this interval.

Table 2: SI-SNRi (dB) of the switching protocol. Higher is better. Avg. denotes the full-utterance average, while ≤ 1 s and > 1 s denote the mean over 200 ms windows after the switch.
<table><tr><td rowspan="2"></td><td rowspan="2">No switch</td><td colspan="3">Known face</td><td colspan="3">New face</td></tr><tr><td>Avg.</td><td>≤ 1s</td><td>&gt;1s</td><td>Avg.</td><td>≤ 1s</td><td>&gt;1s</td></tr><tr><td>AV-SkiM [15]</td><td>9.56</td><td>8.05</td><td>8.50</td><td>9.63</td><td>4.73</td><td>1.08</td><td>10.82</td></tr><tr><td>Tap2Listen (Ours)</td><td>9.51</td><td>9.08</td><td>9.31</td><td>9.79</td><td>8.16</td><td>9.16</td><td>14.38</td></tr></table>

## 3.4 On-device Latency

We export the streaming path as a static-shape ONNX graph and evaluate it on a Galaxy S22 (Snapdragon 8 Gen 1) using ONNX Runtime CPU with fp32, single-thread execution on a fixed big core. After a five-minute thermally settled run, Tap2Listen processes each 8 ms frame in 3.2 ms (40% of the 8 ms budget) at p95, with no over-budget frame. A tap-triggered re-projection takes 5.2 ms at p95, giving 8.5 ms transition latency including the current frame computation. Because regular frames retain substantial timing margin, the additional tap computation is absorbed without accumulating delay. Larger blocks reduce computation but increase buffering delay, making 8 ms the minimum-delay operating point. The reported latency covers the full streaming path, including STFT/iSTFT, frame buffering, encoder and cache updates, target-conditioned mask estimation, and overlap-add reconstruction.

## 4 Conclusion

We proposed Tap2Listen, a streaming AV-TSE model for interactive target switching that separates persistent target-agnostic audio processing from target-conditioned decoding. Tap2Listen confines all target-dependent computation to a single memory projection, so each switch is one re-projection burst (5.2 ms at p95 on a Galaxy S22) whose output matches extraction conditioned on the new speaker from the start to ≈ −70 dB, with no state-driven recovery period. Beyond conventional extraction metrics, we evaluated switching through response delay and post-switch recovery; Tap2Listen achieves 9.68 dB SI-SNRi on LRS3-2mix and substantially improves transition behavior over AV-SkiM. The streaming path also operates within the real-time budget of a single mobile CPU core. These results establish target switching as a distinct systems property of streaming AV-TSE that should be explicitly designed and evaluated at the transition itself.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25602529). This work was also supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (IITP-2026-RS-2026-25615334, AI Star Fellowship Support Program; RS-2019-II190421, Artificial Intelligence Graduate School Program (Sungkyunkwan University)).

## References

[1] Triantafyllos Afouras, Joon Son Chung, and Andrew Zisserman. LRS3-TED: a large-scale dataset for visual speech recognition. CoRR, abs/1809.00496, 2018.

[2] Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age. In 13th IEEE International Conference on Automatic Face & Gesture Recognition, FG 2018, Xi’an, China, May 15-19, 2018, pages 67–74. IEEE Computer Society, 2018.

[3] Honglie Chen, Rodrigo Mira, Stavros Petridis, and Maja Pantic. RT-LA-VocE: Real-Time Low-SNR Audio-Visual Speech Enhancement. In Interspeech 2024, pages 2215–2219, 2024.

[4] Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson, Avinatan Hassidim, William T. Freeman, and Michael Rubinstein. Looking to listen at the cocktail party: a speaker-independent audio-visual model for speech separation. ACM Transactions on Graphics, 37(4), 2018.

[5] Ruohan Gao and Kristen Grauman. Visualvoice: Audio-visual speech separation with cross-modal consistency. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15495–15505, June 2021.

[6] Guilin Hu, Malek Itani, Tuochao Chen, and Shyamnath Gollakota. Proactive hearing assistants that isolate egocentric conversations. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 25366–25383, 2025.

[7] Chen-Yu Lee, Saining Xie, Patrick W. Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-supervised nets. In Guy Lebanon and S. V. N. Vishwanathan, editors, Proceedings ofthe Eighteenth International Conference on Artificial Intelligence and Statistics, AISTATS 2015, San Diego, California, USA, May 9-12, 2015, volume 38 of JMLR Workshop and Conference Proceedings. JMLR.org, 2015.

[8] Junjie Li, Wenxuan Wu, Shuai Wang, Zexu Pan, Kong Aik Lee, Helen Meng, and Haizhou Li. Memo: Attentional momentum for real-time audio-visual target speaker extraction under impaired visual conditions. IEEE Transactions on Audio, Speech and Language Processing, 34:3491–3504, 2026.

[9] Junjie Li, Ke Zhang, Shuai Wang, Kong Aik Lee, Man-Wai Mak, and Haizhou Li. MoMuSE: Momentum multi-modal target speaker extraction for real-time scenarios with impaired visual cues. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6, 2025.

[10] Kai Li, Fenghua Xie, Hang Chen, Kexin Yuan, and Xiaolin Hu. An audio-visual speech separation model inspired by cortico-thalamo-cortical circuits. IEEE Transactions on Pattern Analysis and Machin Intelligence, 46(10):6637–6651, 2024.

[11] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

[12] Teng Ma, Sile Yin, Li-Chia Yang, and Shuo Zhang. Real-Time Audio-Visual Speech Enhancement Using Pre-trained Visual Representations. In Interspeech 2025, pages 61–65, 2025.

[13] Tsubasa Ochiai, Marc Delcroix, Keisuke Kinoshita, Atsunori Ogawa, and Tomohiro Nakatani. Multimodal SpeakerBeam: Single Channel Target Speech Extraction with Audio-Visual Speaker Clues. In Interspeech 2019, pages 2718–2722, 2019.

[14] Zexu Pan, Ruijie Tao, Chenglin Xu, and Haizhou Li. Muse: Multi-modal target speaker extraction with visual cues. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2021, Toronto, ON, Canada, June 6-11, 2021, pages 6678–6682. IEEE, 2021.

[15] Zexu Pan, Wupeng Wang, Shengkui Zhao, Chong Zhang, Kun Zhou, Yukun Ma, and Bin Ma. Online Audio-Visual Autoregressive Speaker Extraction. In Interspeech 2025, pages 1928–1932, 2025.

[16] Samuel Pegg, Kai Li, and Xiaolin Hu. RTFS-net: Recurrent time-frequency modelling for efficient audio-visual speech separation. In The Twelfth International Conference on Learning Representations, 2024.

[17] A.W. Rix, J.G. Beerends, M.P. Hollier, and A.P. Hekstra. Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No.01CH37221), volume 2, pages 749–752 vol.2, 2001.

[18] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, and John R. Hershey. SDR - half-baked or well done? In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2019, Brighton, United Kingdom, May 12-17, 2019, pages 626–630. IEEE, 2019.

[19] Wendi Sang, Kai Li, Runxuan Yang, Jianqiang Huang, and Xiaolin Hu. A fast and lightweight model for causal audio-visual speech separation. In ECAI 2025 — 28th European Conference on Artificial Intelligence, volume 413 of Frontiers in Artificial Intelligence and Applications, pages 509–516. IOS Press, 2025.

[20] Joon Son Chung, Andrew Senior, Oriol Vinyals, and Andrew Zisserman. Lip reading sentences in the wild. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[21] Cees H. Taal, Richard C. Hendriks, Richard Heusdens, and Jesper Jensen. An algorithm for intelligibility prediction of time-frequency weighted noisy speech. IEEE Trans. Speech Audio Process., 19(7):2125–2136, 2011.

[22] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 32–42. IEEE, 2021.

[23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017.

[24] Bandhav Veluri, Justin Chan, Malek Itani, Tuochao Chen, Takuya Yoshioka, and Shyamnath Gollakota. Real-time target sound extraction. In IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023, pages 1–5. IEEE, 2023.

[25] Bandhav Veluri, Malek Itani, Justin Chan, Takuya Yoshioka, and Shyamnath Gollakota. Semantic hearing: Programming acoustic scenes with binaural hearables. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23), pages 89:1–89:15, 2023.

[26] Bandhav Veluri, Malek Itani, Tuochao Chen, Takuya Yoshioka, and Shyamnath Gollakota. Look once to hear: Target speech hearing with noisy examples. In Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’24), pages 37:1–37:16. ACM, 2024.

[27] Cheng Yu, Vahid Ahmadi Kalkhorani, Buye Xu, and DeLiang Wang. Online AV-CrossNet: a Causal and Efficient Audiovisual System for Speech Enhancement and Target Speaker Extraction. In Interspeech 2025, pages 2985–2989, 2025.

[28] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

## A Related Work

## A.1 Non-Streaming Methods

AV-TSE methods use visual cues from on-screen speakers to estimate the target speech from a mixture. Looking to Listen [4] takes the visual stream of each detected face and the mixed audio as input and separates the speech corresponding to each face. The model uses visual information to improve separation performance and to associate each separated speech signal with the corresponding visible speaker. The authors note that the method does not operate in real time and is better suited to video post-processing.

Subsequent studies distinguish the roles of visual cues more explicitly or reuse target speech obtained during separation as an additional speaker cue. VisualVoice [5] uses two distinct visual cues: lip motion and facial appearance. Lip motion provides visual information temporally aligned with the current utterance, while facial appearance from a single image serves as a cue to the speaker’s vocal characteristics. VisualVoice jointly learns speech separation and cross-modal face-voice representations to model this relationship. MuSE [14] reuses target speech from the preceding extraction stage. It generates a self-enrolled speaker embedding from this speech and visual features and uses the embedding to condition mask estimation in the next extractor block. Repeating this process across extractor blocks progressively refines the target speech.

Alongside these developments, other studies improve architectures for fusing and processing audio and visual information. CTCNet [10] extracts audio and visual features at multiple temporal resolutions and repeatedly fuses them at multiple processing stages. The model feeds the fused features back into subsequent audio and visual feature extraction, so each modality influences later stages. RTFS-Net [16] represents audio in the time-frequency domain, models the temporal and frequency dimensions separately, and fuses visual features with the audio representation through attention. It also reduces computational cost by performing its main computations on representations with fewer channels and lower time-frequency resolution. These methods evaluate separation performance with a fixed target speaker in non-streaming settings and do not evaluate user-initiated target switching during processing.

## A.2 Streaming Methods

Streaming methods remove or limit dependence on future inputs to operate in real time. RT-LA-VocE [3] uses causal audio and visual encoders, an Emformer, and a causal vocoder to reduce algorithmic latency to one video frame (40 ms) without future inputs. Online AV-CrossNet [27] uses one-frame look-ahead for online processing, while RAVEN [12] uses a pretrained visual encoder with two-frame look-ahead and runs in real time on a CPU. Swift-Net [19] uses causal convolutions and a unidirectional recurrent module in a lightweight architecture that operates without future inputs.

Some streaming methods reuse information from earlier segments to guide ongoing target speaker extraction. MoMuSE [9] stores past voiceprints of the target speaker in a memory bank and combines them with current speaker information to track the same speaker when visual cues are unreliable. However, the authors evaluate MoMuSE under corrupted or missing visual cues for a fixed target speaker and do not evaluate target switching during streaming. The algorithmic or processing latencies reported in prior streaming studies refer to delays during continuous processing with a fixed target speaker, not the delay until the output responds to a new target speaker after a target switch.

Pan et al. [15] directly evaluate target speaker switching. They switch the target at an arbitrary point while two speakers speak simultaneously and measure SI-SNRi before and after the switch. The authors also argue that aggregating past separated signals into a single speaker embedding can create momentum that hinders rapid target switching. However, average SI-SNRi before and after the switch does not show how separation performance changes immediately after the switch. The proactive hearing assistant [6] measures SI-SDRi over successive intervals after a conversation partner starts speaking, but does not use explicit target selection as the reference point. We use the target-selection time as the reference point and measure both the delay until the output responds to the new target speaker and the temporal evolution of separation performance immediately after the switch.