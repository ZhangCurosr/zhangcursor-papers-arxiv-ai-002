# SPOT, SEPARATE, AND ENHANCE: FULLY GENERATIVE APPROACH FOR AUDIO MIXING

Ilpo Viertola<sup>1,2</sup>, Giulio Cengarle<sup>1</sup>, Gouthaman KV<sup>1</sup>, Daniel Arteaga<sup>1</sup>, Lie Lu<sup>1</sup>

<sup>1</sup>Dolby Laboratories <sup>2</sup>Tampere University

## ABSTRACT

We introduce Spot, Separate, and Enhance (SSE), the first multimodal, user-guided generative model for audio remixing and enhancement. SSE enhances video content by rebalancing the audio, removing unwanted audio sources, and reducing reverberation, guided by both video and textual descriptions. To support its training and evaluation, we propose DegradedMix, a new dataset built on the audio remixing benchmark MuddyMix. We also adopt evaluation metrics from generative modeling, which better capture the creative nature of remixing than standard reconstruction-based metrics. SSE outperforms existing baselines in both controllability and remixing quality, as shown by extensive experiments. Project page: sse-ai.notion.site

Index Terms— Generative Learning, Audio Remixing, Multimodal Learning

## 1. INTRODUCTION

Recent advances in deep learning and generative modeling have opened new possibilities for enhancing User Generated Content (UGC). A large portion of UGC consists of videos captured in everyday environments, where audio often contains multiple sound sources, background noise, reverberation, and other recording artifacts. Despite improvements in video capture hardware, post-processing remains essential for improving audio quality and intelligibility. For example, rebalancing the audio mix to emphasize the primary subject, suppressing background noise, or mitigating reverberation can substantially improve the viewing experience.

To enable automatic audio remixing, Huang et al. [1] introduced Visually Guided Acoustic Highlighting (VGAH), which rebalances the audio mix to emphasize the main subject based on visual information. While an important step toward automated audio enhancement, VGAH primarily considers loudness adjustments and does not address other common degradations such as background noise or reverberation. Together with the task, Huang et al. [1] proposed an evaluation framework based primarily on differences between the generated remix and ground-truth balanced mix in waveform or spectral representations. However, audio remixing is inherently creative, with multiple perceptually valid solutions, and reconstruction-based evaluation may therefore penalize valid outputs that differ from a particular ground truth. These limitations motivate a more general remixing framework that can address diverse degradations and acoustic conditions while allowing greater flexibility in the solution space.

Existing approaches [1, 2] use an encoder-decoder model [3] where a Transformer [4] processes the latent audio representation conditioned on video frames, and the decoder generates the remixed audio. Huang et al. [1] propose VisAH, which is trained end-to-end in a discriminative manner using a reconstruction loss between the automatic remix and the ground truth balanced mix. Malard et al. [2] propose VisAH-FM, a concurrent work that adopts the same architecture with VisAH but replaces the discriminative training objective with a generative Conditional Flow Matching (CFM) [5] formulation. They further introduce a rollout loss [6, 7], which performs full generation during training and computes an Mean Squared Error (MSE) against the ground-truth target, exposing the model to its own intermediate predictions and encouraging self-correction.

However, both approaches have limitations. First, VisAH inherently struggles to generate new audio as it predicts a spectral mask applied to the input mixture. When the target source is heavily buried in noise or reverberation, the model cannot generate new content to replace the degraded audio. Second, both approaches train the audio codec jointly with the generative backbone which limits scalability [8, 9]. Third, neither approach provides user control beyond the video conditioning. Finally, the rollout loss in VisAH-FM introduces overhead by requiring full generation for every training sample.

To address these limitations, we propose Spot, Separate, and Enhance (SSE), a multimodal, user-guided generative framework for audio remixing. Unlike existing approaches that primarily modify the input mixture, SSE can generate new audio to replace severely degraded target-source regions, while explicit user guidance enables control over the desired remixing behavior.

We also introduce a new evaluation framework based on principles from generative modeling, which are better suited to a creative task such as audio remixing. Rather than requiring close reconstruction of a single ground-truth target, our evaluation considers the perceptual quality and effectiveness of the resulting remix. We further define a controlled set of audio degradations to simulate common real-world recording defects, enabling systematic evaluation under challenging acoustic conditions. Extensive experiments demonstrate that SSE achieves superior remixing quality compared to existing approaches while providing greater user controllability and the ability to recover target-source content under severe audio degradation.

## 2. METHOD

SSE, presented in Fig. 1, is a multimodal generative approach for UGC enhancement. Given a video from an everyday environment, SSE enhances the audio according to the user’s preferences. All modalities are first encoded separately. Then, audio and visual features are temporally aligned and fused. The fused features are passed to a generative model, where text guides generation via crossattention, and the output is decoded into a waveform. SSE can rebalance the audio mix, remove unwanted sources, and generate new audio content to replace degraded audio.

## 2.1. Neural Audio Codec – SkipDACVAE

We propose SkipDACVAE, adapted from the DACVAE [10, 11] audio codec. DACVAE [10] replaces the Residual Vector Quantization (RVQ) bottleneck of the original DAC [11] with a Variational

![](images/8e1a23a1ce81e574c06b11ad11c8a17785f2204f6825fb96d6849128c64d9e82.jpg)  
Fig. 1: Overview of SSE. Given a video with degraded and unbalanced audio, SSE enhances it according to the given textual description. First, all the modalities are encoded with specific encoders, and the visual and audio features are aligned and fused. The fused and textual features are then passed to the Diffusion Transformer (DiT). The DiT generates enhanced audio that closely follows the original content and user guidance. Finally, the generated audio is decoded into a waveform representation with the SkipDACVAE decoder.

Autoencoder (VAE) bottleneck. This gives DACVAE a continuous latent space which is well suited for continuous flow matching [5].

SkipDACVAE adds skip connections from the encoder to the decoder. This improves reconstruction quality, since the decoder can access the original audio features directly. Each skip connection adds encoder features to decoder features at matching temporal resolution. Before addition, a 1D convolutional layer with kernel size of 1 maps the encoder features to the decoder’s channel dimension. Each con nection uses its own convolutional layer. The number of connections equals the number of downsampling layers in the encoder.

We initialize SkipDACVAE with pretrained DACVAE weights and freeze all shared layers, leaving only the added 1D convolutional layers trainable. These layers are zero-initialized, so SkipDACVAE behaves identically to DACVAE at the start of training. During training, we obtain a degraded version of the input audio and pass it through SkipDACVAE to compute the skipped values. We then encode the original, undegraded audio and decode it using skipped values computed from corresponding degraded version. We compute the reconstruction loss between the decoded output and the original audio, and backpropagate it to update SkipDACVAE. Using the degraded audio to compute skipped values forces SkipDACVAE to transfer high-level information from encoder to decoder, rather than simply copying the original audio features. We use the original training configuration of DAC [11].

During training and inference of SSE, we use a pretrained Skip-DACVAE. We encode the input audio mixture, yielding a latent sequence $\pmb { x } _ { m i x } \in \mathbb { R } ^ { T \times C _ { a u d } }$ at 25 Hz, where T is the number of audio frames and $C _ { a u d } = 1 2 8$ is the channel dimension.

## 2.2. Visual Encoder

We utilize a pretrained Perception Encoder (PE) [12] as the visual encoder. PE’s large scale contrastive training on vision-language pairs allows it to learn semantically rich representations for actions and scene context. We encode the video per-frame, yielding a sequence of visual features $\pmb { x } _ { v i s } \in \mathbb { R } ^ { C _ { v i s } \times T _ { v i s } }$ , where $T _ { v i s }$ is the number of visual frames and $C _ { v i s } = 1 0 2 4$ is the channel dimension.

## 2.3. Text Encoder

Text descriptions are encoded with a pretrained T5-base encoder [13]. Text feature sequence $\pmb { x } _ { t e x t } \in \mathbb { R } ^ { N \times \bar { C } _ { t e x t } }$ , where N is the number of text tokens and $C _ { t e x t } = 7 6 8$ is the channel dimension, is obtained from the last hidden layer of the encoder. Before feeding the text features to the cross-attention layers of the generative model, we project them to the model’s channel dimension C using a linear layer.

## 2.4. Generative Conditional Flow Matching Model

We propose a generative approach for audio remixing based on CFM [5] and recent audio source separation techniques [14]. Our model learns a continuous vector field that transports a Gaussian prior sample $\pmb { x } _ { 0 } \sim N ( 0 , \pmb { I } )$ to a target data sample ${ \bf x } _ { 1 }$ over $t \in [ 0 , 1 ]$ , where t is the timestep. At each timestep, the model predicts a velocity field ${ \pmb v } _ { \theta } ( t , { \pmb C } , { \pmb x } _ { t } )$ , where $C = \{ \pmb { x } _ { m i x } , \pmb { x } _ { v i s } , \pmb { x } _ { t e x t } \}$ is the conditioning signal, $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is the flow state at time t, and θ are the model parameters. By integrating ${ \pmb v } _ { \theta }$ over time, we obtain the final prediction $\hat { \mathbf { x } } _ { 1 }$

Modality Aligner. The target data sample $\pmb { x } _ { 1 } \in \mathbb { R } ^ { T \times 2 C _ { a u d } }$ stacks the target audio ${ \pmb x } _ { t g t }$ and the residual signal ${ \pmb x } _ { m i x } - { \pmb x } _ { t g t }$ along the channel dimension. The Gaussian prior sample x<sub>0</sub> $\in \mathbb { R } ^ { T \times 2 \bar { C } _ { a u d } }$ stacks noise $z \sim N ( 0 , I )$ and an all-zero tensor in the same way. Following [14], we condition the model on the audio mixture $\mathbf { x } _ { m i x }$ for content-faithful generation. At each flow step, we concatenate the flow state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ with the fixed mixture ${ \pmb x } _ { m i x } .$ , then project to the model’s channel dimension C using a linear layer, yielding $\pmb { h } \in \mathbb { R } ^ { T \times C }$

We also add visual information to h. First, to temporally align the audio and visual features, we match the frame rate of $\pmb { x } _ { v i s } \mathrm { t o } 2 5$ Hz using nearest-neighbor interpolation, yielding a visual feature sequence $\mathbf { \bar { x } } _ { v i s } \in \mathbb { R } ^ { C _ { v i s } \times T }$ . Then, visual information is projected and added via a gated summation: $\pmb { h } = \pmb { h } + \operatorname { t a n h } ( \sigma ) \odot \operatorname { L N } ( \operatorname { C o n v } 1 \mathrm { D } ( \pmb { x } _ { v i s } ) ^ { \top } )$ ), where σ is a learnable gating parameter, ⊙ denotes elementwise (broadcasted) multiplication, LN is layer normalization, and Conv1D is a 1D convolutional layer with kernel size of 1, that projects the visual features to the model’s channel dimension C.

Diffusion Transformer. We represent the velocity field with a DiT [15] and previously introduced Modality aligner. SSE adopts an architecture where each Transformer [4] block is modulated by the flow time embedding via scale-and-shift operations on normalization and residual layers. To create the flow time embedding, a shared MLP maps t to six modulation parameters (four scales, two biases), reused across blocks with layer-specific biases added for depth-dependent effects, reducing model size.

In practice, we sample t as $t ~ = ~ k / ( K - 1 )$ , where $k \sim$ $\mathcal { U } \{ 0 , \dots , K - 1 \}$ and $K \ = \ 4$ As shown in previous works [16, 17, 18, 19], selecting a few timesteps during training may be beneficial during inference time in image-to-image generation tasks. Also, a concurrent work by Malard et al. [2] utilize a similar timestep sampling approach. $K = 4$ yielded the best overall results in our testing, and we use this value for all experiments. During inference, we use Euler integration with the same amount of steps.

We initialize the model with pretrained weights of SAM-Audio [14]. We argue that pretraining the model on a large-scale audio source separation task is beneficial for remixing and enhancement, as it allows the model to learn general audio representations and source separation capabilities. In our experiments, training the query, key, value, and output projections across all DiT blocks together with the Modality aligner and the linear layer projecting the $\mathbf { \mathcal { x } } _ { t e x t }$ features (Section 2.3) yielded the best results. We compared different selective unfreezing strategies, LoRA [20] and full finetuning.

## 3. EXPERIMENTS

## 3.1. Dataset – DegradedMix

We base our data on the MuddyMix dataset [1], also used by the compared approaches [1, 2]. MuddyMix consists of 15,078/1,927/1,789 train/validation/test 10-second clips extracted from movies [21], each with a professionally mixed audio track. Each track is separated into stems via a pretrained source separation model [22], $s = \hat { s } _ { h } +$ $\hat { s } _ { m } + \hat { s } _ { e } + \hat { s } _ { r }$ , denoting speech, music, effects, and residual signals. MuddyMix degrades these stems with random discrete gain changes, then remixes them into a degraded audio track.

To mimic real-world UGC enhancement scenarios, we propose DegradedMix which extends MuddyMix by adding background noise [23] and reverberation [24] to the audio mixes. We train on DegradedMix and evaluate on both DegradedMix (Tables 1 and 2) and MuddyMix (Tables 3 and 4) test sets, enabling comparison to prior work under standard and more challenging degradations. Note that MuddyMix [1] does not support background noise removal, as all the stems in the degraded mix are also present in the ground-truth mix.

We also add audio-based captions to the DegradedMix dataset, which describe the original, non-degraded audio content. One-to-two sentence audio captions are generated automatically with Qwen2- Audio [25]. This mimics the real-world scenario where the user provides a textual description of the desired enhancement. We release the code to generate DegradedMix data.

## 3.2. Compared Methods

The proposed model is compared against two recent approaches, namely VisAH [1] and VisAH-FM [2]. VisAH is a discriminative approach that uses a Transformer-based [4] encoder-decoder model to predict a mask in spectral space which is applied to the input audio to generate the remixed audio. VisAH-FM is a closed-source concurrent work that was published after the original VisAH which uses a conditional flow matching approach to generate the remixed audio, following a similar architecture as VisAH.

## 3.3. Implementation and Training Details

We train on 25 FPS video with 48 kHz audio, resampled to 44.1 kHz for evaluation to match prior work [1, 2]. Video frames are resized to 336 ×336, and SSE uses a 10-second context length, following [1, 2]. We generate degraded audio on-the-fly during training, applying gain changes to the stems and adding random background noise and reverberation. The model is initialized from pretrained weights [14] and trained for approximately 67K steps with batch size 26 on two NVIDIA RTX PRO 6000 Blackwell GPUs, using AdamW [26] $( \beta = [ 0 . 9 , 0 . 9 5 ]$ , weight decay 0.1, learning rate $1 e ^ { - 4 } )$ with a cosine annealing schedule and 2K warmup steps.

## 3.4. Evaluation Metrics

The evaluation protocol proposed by Huang et al. [1] is largely based on comparing the automatic remix with the ground truth balanced mix. These metrics include a) magnitude [27] and envelope [28] distance between the mixes, b) time alignment using Wasserstein distance, and c) semantic alignment using Kullback-Leibler Divergence (KL) [8, 29, 30] and ImageBind Score (IB) [31, 32]. In particular, a) and b) are ill-suited for creative tasks like remixing, where multiple valid solutions can exist. We instead adopt evaluation metrics from generative modeling, which better suit the setting.

Following evaluation practices in multimodal audio generation [8, 9], we propose to use the following metrics to evaluate the remixing performance: a) Frechet Audio Distance (´ FD) [33] and Kullback-Leibler Divergence (KL) [8, 29] for distribution matching, c) Inception Score (IS) [34] to measure audio quality, d) Image-Bind Score (IB) [31] for semantic alignment, e) Audio-Visual Synchronization Score (Sync) [35], and f) Semantical Language-Audio Alignment (CLAP) [36]. For distribution matching scores, we utilize PANNs [37] and PaSST [30] pretrained audio classifiers to extract features from the generated remixes and the ground truth mixes. For IS, we use the same PANNs classifier.

## 3.5. Results

Table 1: Proposed metrics calculated for DegradedMix test data. We could not evaluate VisAH-FM [2] as the model is closed-source. \*: retrained with the DegradedMix data for fair comparison.
<table><tr><td rowspan="2">Model</td><td>PANNs</td><td>PaSST</td><td rowspan="2">IS ↑ IB↑ Sync↓CLAP↑</td></tr><tr><td>FD↓KL↓</td><td>FD↓KL↓</td></tr><tr><td>Input</td><td>3.55 0.35</td><td>43.390.263.16 30.4951.80</td><td>40.56</td></tr><tr><td>VisAH* [1]</td><td>2.970.31</td><td>40.630.213.04 31.0851.80</td><td>44.06</td></tr><tr><td>SSE</td><td>1.260.14 </td><td>41.300.123.13 32.4448.67</td><td>45.06</td></tr></table>

Table 2: VGAH [1] metrics calculated for DegradedMix test data. We could not evaluate VisAH-FM [2] as the model is closed-source. \*: retrained with the DegradedMix data for fair comparison.
<table><tr><td>Variant</td><td>Env↓</td><td>Mag ↓</td><td> $\mathrm { W a s \downarrow }$ </td><td> $\mathrm { K L } _ { \mathrm { P a S S T } } \downarrow$ </td></tr><tr><td>Input</td><td>6.79</td><td>23.13</td><td>1.95</td><td>25.65</td></tr><tr><td>VisAH* [1]</td><td>5.16</td><td>15.70</td><td>1.22</td><td>21.08</td></tr><tr><td>SSE</td><td>3.65</td><td>10.87</td><td>0.90</td><td>11.60</td></tr></table>

Audio Enhancement. Table 1 reports model performance on DegradedMix, which includes gain imbalance, background noise, and reverberation as degradations, using the proposed metrics. For a fair comparison, we retrain VisAH [1] on DegradedMix, and also report metrics for the unprocessed degraded Input. We could not evaluate VisAH-FM [2], as its code is not publicly available. Our model outperforms VisAH across all metrics except PaSST [30] FD, highlighting its effectiveness in enhancing degraded audio. The high CLAP and IB scores further show the model’s ability to follow condi tional guidance, which is crucial for user-guided audio enhancement.

Table 2 reports the model performance on DegradedMix using the VGAH metrics [1]. Our model clearly outperforms VisAH.  
Table 3: Proposed metrics calculated for MuddyMix [1] test data. We could not evaluate VisAH-FM [2] as the model is closed-source and the authors do not provide samples for the MuddyMix test set.
<table><tr><td rowspan="2">Model</td><td>PANNs</td><td>PaSST</td></tr><tr><td>FD↓KL↓</td><td>IS ↑ IB↑ Sync↓CLAP ↑ FD↓  $\mathrm { K L \downarrow }$ </td></tr><tr><td>Input</td><td>2.30 0.29</td><td>31.86 0.21 3.11 31.19 51.19 38.30</td></tr><tr><td>VisAH [1]</td><td>1.26 0.18</td><td>18.06 0.11 3.06 31.81 49.71 39.56</td></tr><tr><td>SSE</td><td>1.040.13</td><td>35.780.103.13 32.7447.69 46.78</td></tr></table>

Table 4: VGAH [1] metrics calculated for MuddyMix [1] test data.
<table><tr><td>Model</td><td>Env↓</td><td>Mag ↓</td><td>Was↓</td><td> $\mathrm { K L } _ { \mathrm { P a S S T } } \downarrow$ </td></tr><tr><td>Input</td><td>6.29</td><td>22.69</td><td>1.96</td><td>20.74</td></tr><tr><td>VisAH [1]</td><td>3.38</td><td>9.99</td><td>0.84</td><td>11.37</td></tr><tr><td>VisAH-FM [2]</td><td>2.74</td><td>8.28</td><td>0.63</td><td>9.70</td></tr><tr><td>SSE</td><td>3.46</td><td>9.86</td><td>0.88</td><td>9.61</td></tr></table>

Audio Remixing. Table 3 reports the model performance on MuddyMix, which includes only gain imbalance as a degradation, using the proposed metrics. We could not evaluate VisAH-FM [2] as its code or samples are not available. SSE exceeds all the compared methods across all the metrics except PaSST [30] FD. However, we clearly achieve the best balance between all the metrics.

Table 4 reports model performance on MuddyMix [1] using VGAH metrics. When compared against a single ground-truth solution (Env, Mag, Was), SSE falls short of the concurrent work VisAH-FM [2], though we achieve the best ${ \mathrm { K L } } _ { \mathrm { P a S S T } }$ score. We argue this reflects a limitation of the metric rather than the model: in a creative task like remixing, where multiple valid solutions can exist, comparing generated results to a single ground truth is not ideal. This is further highlighted by the subjective evaluation in Table 5, where our approach (SSE-L) is preferred over VisAH with a large margin, despite scoring only marginally better or on par under VGAH metrics.

Table 5: Pairwise preference test results for UGC enhancement. Preference rate is the proportion of trials in which the first condition was preferred over the second, with 95% confidence intervals. pvalues are from a two-sided binomial test against chance (50%).
<table><tr><td>Comparison (A vs. B)</td><td>Wins (A-B)</td><td>Pref. rate</td><td>p-value</td></tr><tr><td>SSE-L vs. SSE-S</td><td>68-22</td><td>75.6%</td><td>&lt; 0.001</td></tr><tr><td>SSE-L vs. Original audio</td><td>72-18</td><td>80.0%</td><td>&lt; 0.001</td></tr><tr><td>SSE-L vs. VisAH [1]</td><td>80-10</td><td>88.9%</td><td>&lt; 0.001</td></tr><tr><td>SSE-S vs. Original audio</td><td>47-43</td><td>52.2%</td><td>0.752</td></tr><tr><td>SSE-S vs. VisAH [1]</td><td>57-33</td><td>63.3%</td><td>0.015</td></tr><tr><td>Original audio vs. VisAH [1]</td><td>62-28</td><td>68.9%</td><td>&lt; 0.001</td></tr></table>

Subjective Analysis. We conduct a subjective evaluation on UGC audio enhancement. We select 10 UGC videos (presented at the project page) and generate enhanced audio using the small- and largevariants of SSE (SSE-S and SSE-L), and VisAH [1]. 10 participants select the preferred audio for a video between two samples, presented in random order. Preference rate tells how many times on average a particular audio was preferred over other options. Table 5 shows the final rates, where SSE-L is preferred in 81% of the cases, demonstrating its effectiveness. SSE-S is preferred in 47% and the original audio is preferred in 46% of the cases. VisAH’s limited UGC enhancement and remixing capabilities result in a lower preference rate of 26%.

## 3.6. Ablations

Table 6: Ablation on different model variants. We use MuddyMix [1] data. Preferred configuration is highlighted in yellow.
<table><tr><td rowspan="2">Model</td><td>PANNs</td><td>PaSST</td><td rowspan="2">Sync ↓ CLAP ↑</td></tr><tr><td>FD KL↓</td><td>IS↑ IB↑ FD KL↓</td></tr><tr><td>Small</td><td>1.12 0.14</td><td>37.81 0.11 3.13 32.76 47.82</td><td>47.02</td></tr><tr><td>Base</td><td>1.06 0.13</td><td>37.28 0.10 3.12 32.64 48.84</td><td>47.10</td></tr><tr><td>Large</td><td>1.04 0.13 35.78</td><td>0.103.13 32.74 47.69</td><td>46.78</td></tr></table>

Model Size. Table 6 presents the performance of different model sizes. Small-variant has 500M parameters, base-variant has 1B parameters, and large-variant has 3B parameters. The approach is scalable and we observed that the large-variant yields the best performance across the majority of the metrics. We conduct all the experiments using the large-variant of the model if not otherwise specified.

Table 7: Effect of using SkipDACVAE. We use MuddyMix [1] data. Preferred configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td>PANNs</td><td>PaSST</td></tr><tr><td>FD↓KL↓ FD↓</td><td>IS ↑ IB ↑ Sync↓CLAP ↑ KL↓</td></tr><tr><td>DACVAE</td><td>3.380.21 143.55 0.283.03 32.5048.35</td><td>48.13</td></tr><tr><td>SkipDACVAE</td><td>1.04 0.13 35.78 0.10 3.13 32.74 47.69</td><td>46.78</td></tr></table>

SkipDACVAE. As shown in Table 7, adding skip connections boosts the performance significantly. We conduct the experiments using the MuddyMix [1] data. The added representation capacity of the neural audio codec allows for better reconstruction of the audio which is reflected in the improved performance across all metrics. Note that the model is not retrained for the SkipDACVAE as these two neural audio codecs share the same latent space.

Table 8: Ablation on different caption types. We use small-variant and MuddyMix [1]. Preferred configuration is highlighted in yellow.
<table><tr><td rowspan="2">Type</td><td>PANNs PaSST</td></tr><tr><td>IS ↑ IB ↑ Sync ↓CLAP ↑ FD ↓KL↓ FD↓ KL↓</td></tr><tr><td>Original</td><td>1.24 0.16 39.150.12 3.13 32.5849.97 44.82</td></tr><tr><td>Summarized</td><td>1.22 0.16 38.470.123.12 32.6649.11 44.78</td></tr><tr><td></td><td>Audio-based 1.12 0.14 37.81 0.11 3.13 32.76 47.82 47.02</td></tr></table>

Captions. We ablate on different styles of textual descriptions in Table 8 and use the small-variant of the model with MuddyMix [1]. MuddyMix dataset provides captions for each video frame, extracted with 1 FPS. These Original captions are long and contain unrelated information for audio enhancement. Summarized captions are generated by summarizing the original ones using Claude Sonnet 5 [38]. Finally, Audio-based captions are generated by describing the original, non-degraded, audio content of the video using Qwen2-Audio [25]. We use audio-based captions, as they mimic the real-world scenario where users provide a description of the desired enhancement.

## 4. CONCLUSION

SSE shows that a single multimodal generative model can enhance UGC audio. It outperforms VisAH in controllability and quality under both generative and reconstruction-based metrics, though results are more mixed under single-ground-truth metrics, underscoring the need for evaluation that embraces remixing’s one-to-many nature. We hope DegradedMix and our evaluation framework support this shift.

## 5. REFERENCES

[1] C. Huang, R. Gao, J.M.F. Tsang, J. Kurcius, C. Bilen, C. Xu, A. Kumar, and S. Parekh, “Learning to highlight audio by watching movies,” in CVPR, 2025.

[2] H. Malard, G.L. Lan, D. Wong, D.L. Lou Alon, Y. Wu, and S. Parekh, “Conditional flow matching for visually-guided acoustic highlighting,” arXiv:2602.03762, 2026.

[3] S. Rouard, F. Massa, and A. Defossez, “Hybrid transformers´ for music source separation,” in ICASSP, 2023.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in NeurIPS, 2017.

[5] Y. Lipman, R. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in ICLR, 2023.

[6] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in ICLR, 2023.

[7] Y. Song and P. Dhariwal, “Improved techniques for training consistency models,” in ICLR, 2024.

[8] H. Liu, Y. Yuan, X. Liu, X. Mei, Q. Kong, Q. Tian, Y. Wang, W. Wang, Y. Wang, and M. D. Plumbley, “AudioLDM 2: Learning holistic audio generation with self-supervised pretraining,” TASLP, 2024.

[9] H. K. Cheng, M. Ishii, A. Hayakawa, T. Shibuya, A. Schwing, and Y. Mitsufuji, “MMAudio: Taming multimodal joint training for high-quality video-to-audio synthesis,” in CVPR, 2025.

[10] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C. Ma, C. Chuang, et al., “Movie Gen: A cast of media foundation models,” arXiv:2410.13720, 2024.

[11] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved RVQGAN,” in NeurIPS, 2023.

[12] D. Bolya, P. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed, et al., “Perception encoder: The best visual embeddings are not at the output of the network,” arXiv:2504.13181, 2025.

[13] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” JMLR, 2020.

[14] B. Shi, A. Tjandra, J. Hoffman, H. Wang, Y.-C. Wu, L. Gao, J. Richter, M. Le, A. Vyas, S. Chen, C. Feichtenhofer, P. Dollar,´ W.-N. Hsu, and A. Lee, “SAM audio: Segment anything in audio,” arXiv:2512.18099, 2025.

[15] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in ICCV, 2023.

[16] C. Chadebec, O. Tasar, S. Sreetharan, and B. Aubin, “LBM: Latent bridge matching for fast image-to-image translation,” in ICCV, 2025.

[17] C. Chadebec, O. Tasar, E. Benaroche, and B. Aubin, “Flash diffusion: Accelerating any conditional diffusion model for few steps image generation,” in AAAI, 2025.

[18] S. Luo, Y. Tan, L. Huang, J. Li, and H. Zhao, “Latent consistency models: Synthesizing high-resolution images with fewstep inference,” arXiv:2310.04378, 2023.

[19] T. Salimans and J. Ho, “Progressive distillation for fast sampling of diffusion models,” in ICLR, 2022.

[20] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in ICLR, 2022.

[21] M. Bain, A. Nagrani, A. Brown, and A. Zisserman, “Condensed movies: Story based retrieval with contextual embeddings,” in ACCV, 2020.

[22] K. N. Watcharasupat, C. Wu, and I. Orife, “Remastering Divide and Remaster: A cinematic audio source separation dataset with multilingual support,” in 5th IEEE International Symposium on the Internet of Sounds (IS2), 2024.

[23] G. Wichern, J. Antognini, M. Flynn, L. R. Zhu, E. McQuinn, D. Crow, E. Manilow, and J. Le Roux, “Wham!: Extending speech separation to noisy environments,” in Interspeech, 2019.

[24] C. Bagwell and SoX Contributors, “SoX - Sound eXchange,” http://sox.sourceforge.net, 2015, Version 14.4.2.

[25] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, C. Zhou, and J. Zhou, “Qwen2-Audio technical report,” arXiv:2407.10759, 2024.

[26] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.

[27] X. Xu, H. Zhou, Z. Liu, B. Dai, X. Wang, and D. Lin, “Visually informed binaural audio generation without binaural audios,” in CVPR, 2021.

[28] S. Liang, C. Huang, Y. Tian, A. Kumar, and C. Xu, “Avnerf: Learning neural fields for real-world audio-visual scene synthesis,” in NeurIPS, 2023.

[29] A. Vyas, B. Shi, M. Le, A. Tjandra, Y. Wu, B. Guo, J. Zhang, X. Zhang, R. Adkins, W. Ngan, et al., “AudioBox: Unified audio generation with natural language prompts,” arXiv:2312.15821, 2023.

[30] K. Koutini, J. Schluter, H. Eghbal-zadeh, and G. Widmer, “Ef-¨ ficient training of audio transformers with patchout,” in Interspeech, 2022.

[31] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “ImageBind one embedding space to bind them all,” in CVPR, 2023.

[32] I. Viertola, V. Iashin, and E. Rahtu, “Temporally aligned audio for video with autoregression,” in ICASSP, 2025.

[33] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Frechet ´ audio distance: A metric for evaluating music enhancement algorithms,” arXiv:1812.08466, 2018.

[34] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training GANs,” in NeurIPS, 2016.

[35] V. Iashin, W. Xie, E. Rahtu, and A. Zisserman, “Synchformer: Efficient synchronization from sparse cues,” in ICASSP, 2024.

[36] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in ICASSP, 2023.

[37] Q. Kong, Yn Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” TASLP, 2020.

[38] Anthropic, “Claude Sonnet 5,” https://www.anthropic. com/claude, 2026, Large language model.