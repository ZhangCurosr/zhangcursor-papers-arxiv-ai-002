# MeanVoiceFlow2: Joint Optimization of Mean Flow and Content Encoder for Fast One-Step Zero-Shot Voice Conversion

Takuhiro Kaneko <sup>ID</sup> , Hirokazu Kameoka <sup>ID</sup> , Kou Tanaka <sup>ID</sup> , Yuto Kondo <sup>ID</sup>

NTT, Inc., Japan

takuhiro.kaneko@ntt.com

## Abstract

Flow-matching approaches to voice conversion (VC) have gained attention owing to their high speech quality and strong speaker similarity. Among them, one-step models such as MeanVoiceFlow are particularly attractive because they enable efficient inference; however, their reliance on a computationally intensive content encoder remains a bottleneck. We therefore propose MeanVoiceFlow2, a framework that jointly optimizes a flow-based conversion module and a computationally efficient content encoder. The model is trained through conversion distillation using MeanVoiceFlow and the reconstruction of real data. We further incorporate diffusion-GAN training with sample mixing and teacher-guided conditioning augmentation to enhance realism and disentanglement. Experiments on zeroshot VC showed that MeanVoiceFlow2 achieved higher perceptual quality and approximately 9× faster inference than Mean-VoiceFlow while maintaining comparable speaker similarity. Index Terms: zero-shot voice conversion, flow matching, knowledge distillation, adversarial training, efficient inference

## 1. Introduction

Voice conversion (VC) transforms the voice of a source speaker into that of a target speaker while preserving the linguistic content. In particular, nonparallel (zero-shot) VC approaches have attracted attention owing to their flexibility in practical scenarios. Although training without explicit pairwise supervision is challenging, recent advances in deep generative models (e.g., [1–5]) have improved performance (e.g., [6–12]). Recent diffusion [13–15] and flow-matching [16–18] models have further advanced VC [19–28], achieving high speech quality and strong speaker similarity. However, most approaches rely on multi-step generation, which results in slow inference.

One-step diffusion- and flow-based methods [29–31] have been proposed to address this issue. These include MeanVoice-Flow [31], an approach based on Mean Flow [32]. Unlike other one-step models [29, 30], MeanVoiceFlow can be trained entirely from scratch without external pretrained modules (e.g., a pretrained neural vocoder). However, despite the acceleration of the main flow process, it still requires a computationally intensive content encoder, which remains a primary computational bottleneck. In practice, the inference time of a content encoder is approximately ten times that of a single flow step.

To address this issue, we propose MeanVoiceFlow2, which jointly optimizes a flow-based conversion module and a content encoder. First, the model is trained through conversion distillation using MeanVoiceFlow [31] to align its behavior with that of the teacher, together with real-data reconstruction to improve generation fidelity. Second, diffusion-GAN training [33] with sample mixing is used to improve the realism without relying on external modules. Third, teacher-guided conditioning augmentation is used to enhance content–speaker disentanglement.

FasterVoiceGrad [34] also jointly optimizes the conversion module and content encoder. However, it relies on a pretrained neural vocoder (e.g., [35]) for stable adversarial training, whereas our method requires no additional pretrained modules beyond the teacher model trained on the same data. Furthermore, FasterVoiceGrad is based on a diffusion formulation, whereas our approach is based on Mean Flow.

Experiments on zero-shot (nonparallel any-to-any) VC demonstrated the effectiveness of each component and showed that MeanVoiceFlow2 achieved higher perceptual quality and significantly faster inference than MeanVoiceFlow while maintaining a comparable speaker similarity.

The remainder of this paper is organized as follows: Section 2 reviews MeanVoiceFlow. Section 3 presents MeanVoice-Flow2. Section 4 presents our experimental results. Finally, Section 5 concludes the paper.

## 2. Preliminary: MeanVoiceFlow

In this section, we briefly review MeanVoiceFlow [31], which serves as the basis of MeanVoiceFlow2. MeanVoiceFlow is a one-step zero-shot VC model based on Mean Flow [32]. For further details, refer to the original paper [31].

MeanVoiceFlow models a mel-spectrogram $x \sim p _ { \mathrm { d a t a } } ( x )$ and performs conversion conditioned on speaker embedding s and content embedding c, which are extracted by a speaker encoder (e.g., [36]) and a content encoder (e.g., [37]), respectively. During the conversion, speaker embedding $s ^ { \mathrm { t g t } }$ is extracted from the target speech and content embedding $c ^ { \mathrm { s r c } }$ is extracted from the source speech. Here, the superscripts src and tgt denote source and target speakers, respectively.

MeanVoiceFlow considers the following linear flow path:

$$
z _ { t } ^ { \mathrm { t g t } } = ( 1 - t ) x _ { \mathrm { r e a l } } ^ { \mathrm { t g t } } + t \hat { \epsilon } ^ { \mathrm { s r c } } ,\tag{1}
$$

where $t \in [ 0 , 1 ]$ is the time. $\hat { \epsilon } ^ { \mathrm { s r c } }$ indicates the diffused source sample and is defined as follows:

$$
\hat { \epsilon } ^ { \mathrm { s r c } } = ( 1 - \alpha ) x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } + \alpha \epsilon ,\tag{2}
$$

where, $\epsilon \sim \mathcal { N } ( 0 , I )$ During training, $\alpha \in [ 0 , 1 ]$ is sampled from a logit-normal distribution [38], that is, $\alpha ^ { \prime } \sim \mathcal { N } ( 0 , 1 )$ and $\alpha = \sigma ( \alpha ^ { \prime } )$ , where $\sigma ( \cdot )$ denotes the sigmoid function. During inference, α is fixed to a constant value. Using $\hat { \epsilon } ^ { \mathrm { s r c } }$ instead of pure noise ϵ yields a unified formulation that covers both unconditional generation and source-conditioned conversion.

(a) Conversion distillation

Let $v ( z _ { t } , t , s , c , \alpha )$ denote the instantaneous velocity at $( z _ { t } , t )$ conditioned on s, c, and α. MeanVoiceFlow employs the average velocity over the interval $[ r , t ]$

$$
u ( z _ { t } , r , t , s , c , \alpha ) = \frac { 1 } { t - r } \int _ { r } ^ { t } v ( z _ { \tau } , \tau , s , c , \alpha ) d \tau ,\tag{3}
$$

which represents the average displacement between r and t.

The average velocity is modeled by a neural network u<sub>θ</sub>. One-step conversion is performed as follows:

$$
\begin{array} { r } { x _ { \theta } ^ { \mathrm { c o n v } } = \hat { \epsilon } ^ { \mathrm { s r c } } - u _ { \theta } \left( \hat { \epsilon } ^ { \mathrm { s r c } } , 0 , 1 , s ^ { \mathrm { t g t } } , c ^ { \mathrm { s r c } } , \alpha \right) , } \end{array}\tag{4}
$$

where $s ^ { \mathrm { t g t } }$ is obtained by randomly shuffling $s ^ { \mathrm { s r c } }$ within a mini-batch during training. For simplicity, we denote u<sub>θ</sub> $( \hat { \epsilon } ^ { \mathrm { s r c } } , 0 , 1 , s ^ { \mathrm { t g t } } , c ^ { \bar { \mathrm { s r c } } } , \alpha )$ by $\dot { u } _ { \theta } ^ { \mathrm { c o n v } }$

## 3. Proposal: MeanVoiceFlow2

In MeanVoiceFlow, the content embedding c is provided as an input to the model. However, extracting c requires a computationally intensive content encoder (e.g., a Conformer [39]-based model [37]), which is the main computational bottleneck, as discussed in Section 1.<sup>2</sup>

To address this issue, we replace the fixed pretrained content encoder, $c _ { \theta } .$ with a computationally efficient and trainable content encoder, $c _ { \phi } ,$ as shown in Figure 1. In addition, we introduce a student average velocity network $u _ { \phi } ,$ , which is jointly optimized with $c _ { \phi }$ under the guidance of the pretrained teacher model u . Here, θ denotes the fixed teacher parameters, and ϕ represents the trainable student parameters. This design enables end-to-end training while eliminating the computational bottleneck caused by a fixed content encoder.

From a training perspective, MeanVoiceFlow2 comprises three key components: (1) joint conversion distillation and realdata reconstruction, (2) diffusion-GAN training with sample mixing, and (3) teacher-guided conditioning augmentation.

## 3.1. Joint conversion distillation and reconstruction

Conversion distillation. We introduce the following conversion distillation loss to mimic the input–output behavior of the teacher model:

$$
\mathcal { L } _ { \mathrm { d i s t } } ^ { \mathrm { c o n v } } ( \phi ) = \mathbb { E } \left[ d \left( u _ { \theta } ^ { \mathrm { c o n v } } , u _ { \phi } ^ { \mathrm { c o n v } } \right) \right] .\tag{5}
$$

Here, $u _ { \theta } ^ { \mathrm { c o n v } }$ and $u _ { \phi } ^ { \mathrm { c o n v } }$ denote the teacher and student outputs, respectively, in the conversion setting, where the teacher output is defined using Eq. 4. The student output is defined as follows:

$$
u _ { \phi } ^ { \mathrm { c o n v } } = u _ { \phi } \left( \hat { \epsilon } ^ { \mathrm { s r c } } , 0 , 1 , s ^ { \mathrm { t g t } } , c _ { \phi } ( x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } ) , \alpha \right) .\tag{6}
$$

The function $d ( \cdot , \cdot )$ denotes the distance measure. Following prior work $[ 3 1 , 3 2 ] .$ , we adopt an adaptively weighted loss [40], $\begin{array} { r } { d ( a , b ) = \frac { \| a - b \| _ { 2 } ^ { 2 } } { \mathrm { s g } \big ( \| a - b \| _ { 2 } ^ { 2 } + \varepsilon \big ) } } \end{array}$ , where sg(·) denotes the stop-gradient operation and $\varepsilon = 1 0 ^ { - 3 }$ is used to prevent division by zero.

Real-data reconstruction. In the distillation framework above, the student is supervised by the teacher-generated conversion output. However, distillation alone does not directly ensure consistency with real data. To address this without relying on paired source–target data in a nonparallel setting, we introduce the following real-data reconstruction loss:

$$
\mathcal { L } _ { \mathrm { r e a l } } ^ { \mathrm { r e c } } ( \phi ) = \mathbb { E } \left[ d \left( u _ { \phi } ^ { \mathrm { r e c } } , u _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } \right) \right] .\tag{7}
$$

![](images/f560816affa4393f0e4ab2bf964d98981839d51e2fd38e3501f048d95da31729.jpg)

![](images/197d907c920a7fabd84127a51bc472a9f166d9fc88252af084fac0f0163dc7f3.jpg)  
Figure 1: Overview of MeanVoiceFlow2. The student jointly learns a computationally efficient content encoder $c _ { \phi }$ and an average velocity network $u _ { \phi }$ through (a) conversion distillation and (b) real-data reconstruction. We further incorporate diffusion-GAN training with sample mixing using a discriminator $\mathcal { D } _ { \psi }$ to promote realism, and teacher-guided conditioning augmentation based on $x _ { \theta } ^ { \mathrm { a u g } }$ to promote disentanglement.

Here, $u _ { \phi } ^ { \mathrm { r e c } }$ denotes the student output under the reconstruction condition, and is defined as follows:

$$
\begin{array} { r } { u _ { \phi } ^ { \mathrm { r e c } } = u _ { \phi } \left( \hat { \epsilon } ^ { \mathrm { s r c } } , 0 , 1 , s ^ { \mathrm { s r c } } , c _ { \phi } ( x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } ) , \alpha \right) . } \end{array}\tag{8}
$$

The ground-truth velocity derived from real source data is

$$
u _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } = \hat { \epsilon } ^ { \mathrm { s r c } } - x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } .\tag{9}
$$

## 3.2. Diffusion-GAN training with sample mixing

Adversarial conversion training. Using only the pairwise distance loss often leads to statistical averaging, which degrades perceptual realism. To address this issue, we introduce adversarial training based on a diffusion-GAN [33], which promotes distribution-level alignment while stabilizing optimization. Additionally, we incorporate data mixing to smooth the discriminator decision boundary and emphasize the subtle discrepancies between the teacher and student samples.

Eq. 4 is used to obtain teacher- and student-generated samples $x _ { \theta } ^ { \mathrm { { { \hat { c } o n v } } } }$ and $x _ { \phi } ^ { \mathrm { { c o n v } } }$ , respectively. The teacher samples are regarded as real examples to guide the student distribution toward the teacher distribution. We then construct mixed samples as follows:

$$
x _ { \mathrm { m i x } } ^ { \mathrm { c o n v } } = ( 1 - \beta ) x _ { \theta } ^ { \mathrm { c o n v } } + \beta x _ { \phi } ^ { \mathrm { c o n v } } ,\tag{10}
$$

where $\beta \in [ 0 , 1 ]$ is drawn from a logit-normal distribution [38] in the same manner as α.

Both the teacher and mixed samples are further perturbed by the diffusion process [33]:

$$
\begin{array} { r } { \hat { \epsilon } _ { \theta } ^ { \mathrm { c o n v } } = ( 1 - \gamma ) x _ { \theta } ^ { \mathrm { c o n v } } + \gamma \epsilon ^ { \prime } , } \\ { \hat { \epsilon } _ { \mathrm { m i x } } ^ { \mathrm { c o n v } } = ( 1 - \gamma ) x _ { \mathrm { m i x } } ^ { \mathrm { c o n v } } + \gamma \epsilon ^ { \prime } , } \end{array}\tag{11}
$$

(12)

where $\epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$ and $\gamma \in [ 0 , 1 ]$ is sampled from a logitnormal distribution.

We adopt a least-squares GAN objective [41],

$$
\mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { c o n v } } ( \psi ) = \mathbb { E } \Big [ ( \mathcal { D } _ { \psi } \big ( \hat { \epsilon } _ { \theta } ^ { \mathrm { c o n v } } , s ^ { \mathrm { t g t } } , c ^ { \mathrm { s r c } } , \gamma \big ) - 1 \big ) ^ { 2 }
$$

$$
+ \left. ( \mathcal { D } _ { \psi } ( \hat { \epsilon } _ { \mathrm { m i x } } ^ { \mathrm { c o n v } } , s ^ { \mathrm { t g t } } , c ^ { \mathrm { s r c } } , \gamma ) ) ^ { 2 } \right] ,\tag{13}
$$

$$
\mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { c o n v } } ( \phi ) = \mathbb { E } \Big [ ( \mathcal { D } _ { \psi } \big ( \hat { \epsilon } _ { \mathrm { m i x } } ^ { \mathrm { c o n v } } , s ^ { \mathrm { t g t } } , c ^ { \mathrm { s r c } } , \gamma \big ) - 1 \big ) ^ { 2 } \Big ] ,\tag{14}
$$

where $\mathcal { D } _ { \psi }$ denotes the discriminator. This GAN objective is conditioned on $s ^ { \mathrm { t g t } } , c ^ { \mathrm { s r c } }$ , and γ to maintain consistency with the conversion setting and diffusion noise level. In practice, the discriminator $\mathcal { D } _ { \psi }$ adopts the same network architecture as the teacher average velocity network u<sub>θ</sub> with time variables r and t omitted.

Adversarial reconstruction training. We apply the same diffusion-GAN procedure with the data-mixing strategy to the reconstructed samples, $x _ { \phi } ^ { \mathrm { r e c } } = \hat { \epsilon } ^ { \mathrm { s r c } } - u _ { \phi } ^ { \mathrm { r e c } }$ , and corresponding real data, $x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } }$ . The mixing and diffusion procedures are illustrated in Figure 1(b). The resulting adversarial losses are denoted as $\mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { r e c } } ( \psi )$ and $\mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { r e c } } ( \phi )$

## 3.3. Teacher-guided conditioning augmentation

VC aims to modify speaker identity while preserving linguistic content, which requires effective content–speaker disentanglement. To encourage the content encoder to learn speakerinvariant representations, we introduce a teacher-guided conditioning augmentation strategy.

The key idea is that the content encoder should produce consistent representations even when speaker identity changes but the underlying linguistic content remains unchanged. To promote this property, we introduce an additional forward path in which the input to the content encoder is replaced with the speech generated by the teacher model under speakeraugmented conditions.

Specifically, we generate augmented samples using the teacher model:

$$
x _ { \theta } ^ { \mathrm { a u g } } = \epsilon ^ { \prime \prime } - u _ { \theta } ( \epsilon ^ { \prime \prime } , 0 , 1 , s ^ { \mathrm { a u g } } , c ^ { \mathrm { s r c } } , 1 ) ,\tag{15}
$$

where $\epsilon ^ { \prime \prime } \sim \mathcal { N } ( 0 , I )$ and $s ^ { \mathrm { a u g } }$ is obtained by randomly shuffling $s ^ { \mathrm { s r c } }$ within a mini-batch.

The resulting representation $c _ { \phi } ( x _ { \theta } ^ { \mathrm { a u g } } )$ replaces $c _ { \phi } \big ( x _ { \mathrm { r e a l } } ^ { \mathrm { s r c } } \big )$ in Eqs. 6 and 8 as shown in Figure 1. Conversion distillation and real-data reconstruction are performed in the same manner as described above. The corresponding losses are denoted as ${ \mathcal { L } } _ { \mathrm { d i s t } } ^ { \mathrm { c o n v - a u g } } ( \phi )$ and $\mathcal { L } _ { \mathrm { r e a l } } ^ { \mathrm { r e c - a u g } } ( \phi )$ , respectively.

Final objective. The final student objective is as follows:

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { \mathrm { M V F 2 } } ( \phi ) = \underbrace { \mathcal { L } _ { \mathrm { d i s t } } ^ { \mathrm { c o n v } } ( \phi ) + \mathcal { L } _ { \mathrm { d i s t } } ^ { \mathrm { c o n v - a u g } } ( \phi ) + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { c o n v } } ( \phi ) } _ { \mathrm { C o n v e r s i o n } } } \\ & { } & { + \underbrace { \mathcal { L } _ { \mathrm { r e a l } } ^ { \mathrm { r e c } } ( \phi ) + \mathcal { L } _ { \mathrm { r e a l } } ^ { \mathrm { r e c - a u g } } ( \phi ) + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { r e c } } ( \phi ) } _ { \mathrm { R e c o n s t r u c t i o n } } , ~ ( } \end{array}\tag{16}
$$

where $\lambda _ { \mathrm { a d v } }$ is a weighting hyperparameter, which had a fixed value of 1 in all of the experiments. The corresponding discriminator objective is expressed as follows:

$$
\mathcal { L } _ { \mathrm { M V F 2 } } ( \psi ) = \mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { c o n v } } ( \psi ) + \mathcal { L } _ { \mathrm { a d v } } ^ { \mathrm { r e c } } ( \psi ) .\tag{17}
$$

## 4. Experiments

## 4.1. Experimental setup

Data. We evaluated MeanVoiceFlow2 on zero-shot (nonparallel any-to-any) VC tasks. The experimental protocol followed that of MeanVoiceFlow [31], which served as the teacher model. Our primary experiments were conducted using the VCTK dataset [42], which contains recordings from 110 English speakers. To examine whether our findings are robust across datasets, we also conducted experiments using LibriTTS [43], which contains recordings from 1,151 English speakers. To simulate unseen-to-unseen conversion, we held out ten speakers and ten sentences from the training for evaluation. All audio clips were downsampled to 22.05 kHz. We extracted 80-dimensional log-mel spectrograms using an FFT size of 1,024, hop size of 256, and window size of 1,024, which served as the conversion targets.

Implementation. To isolate the effects of the proposed training strategy, we adopted the same architectures as that used in previous studies [31, 34]. The velocity networks u and $u _ { \phi }$ were implemented as a U-Net [44] with 12 convolutional layers (512 channels), two down/up-sampling stages, gated linear units (GLUs) [45], and weight normalization (WN) [46]. The discriminator $\mathcal { D } _ { \psi }$ shared the same architecture with the time variables omitted. The proposed content encoder $c _ { \phi }$ comprises three convolutional layers (512 channels), GLUs, instance normalization [47], and WN. For the teacher model, content embeddings were extracted using a pretrained bottleneck feature extractor [37], and speaker embeddings were obtained from a pretrained speaker encoder [36]. The waveforms were synthesized using HiFi-GAN V1 [35]. The teacher and student models were trained using Adam [48] (batch size 32, learning rate $2 \times 1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 5 .$ , and $\beta _ { 2 } = 0 . 9 $ ). The teacher was trained for 500 epochs, and the student and discriminator were trained for 250 epochs, initialized by the pretrained teacher. At the inference time, only $c _ { \phi }$ and $u _ { \phi }$ were used, without the pretrained bottleneck extractor, which eliminated the computational overhead of the heavy fixed content encoder required by the teacher.

Evaluation metrics. The conversion performance was evaluated using five objective metrics. The perceptual quality was measured using three predicted mean opinion score (MOS) metrics: UT↑ (UTMOS [49]) and DNSP↑ (DNSMOS Pro [50], trained on BVCC [51]) for synthesized/converted speech, and DNS↑ (DNSMOS [52]) for noise-suppressed speech. We also used CER↓ with Whisper-large-v3 [53] for intelligibility and SECS↑ with WavLM Base+ [54] for speaker similarity. All of the metrics were computed for 8,100 speaker–sentence pairs. For the key comparisons in Section 4.3, we conducted additional subjective evaluations and speed assessments.

## 4.2. Component analysis

Analysis of joint conversion and reconstruction. We analyzed the effects of joint conversion and reconstruction. Table 1 summarizes the ablations in which the conversion- and reconstruction-related components were removed. Conditioning augmentation was not performed to isolate its effects. (i) Full model (c) achieved the best overall performance across most metrics. (ii) Removing reconstruction (a) degraded perceptual quality (UT, DNSP) and intelligibility (CER), indicating that reconstruction helped align the generated distribution with the real-data distribution beyond teacher guidance. (iii) Removing conversion (b) yielded a low CER but severely degraded speaker similarity (SECS) and perceptual quality (UT, DNSP), which suggests that the model largely preserved the input speech characteristics rather than performing proper VC.

Analysis of adversarial training. The proposed adversarial framework comprises (1) diffusion-GAN training, (2) sample mixing, and (3) no reliance on external modules (e.g., a pretrained neural vocoder), unlike FastVoiceGrad [29, 30], which employs a pretrained vocoder with either a waveform discriminator (WD) [55] or a vocoder-projected feature discriminator (VPFD) [30]. Ablation was performed to evaluate each component. Conditioning augmentation was not performed to isolate its effects. Table 2 summarizes these results. (i) A comparison between (a) and (e) shows that adversarial training improved perceptual quality (DNSP, DNS), intelligibility (CER), and speaker similarity (SECS) over the non-adversarial baseline while maintaining UT. (ii) A comparison between (b) and (c) shows that diffusion improved UT and reduced CER, highlighting its role in preserving speech quality. (iii) A comparison between (b) and (d) shows that sample mixing improved DNSP and DNS, suggesting a regularization effect. (iv) Full configuration (e) achieved the best overall. (v) Compared with WD (f) and VPFD (g), the proposed method (e) achieved competitive or superior performance without a pretrained neural vocoder.

Table 1: Analysis of joint conversion and reconstruction. Conv and Rec indicate the use ofconversion distillation and real-data reconstruction, respectively
<table><tr><td></td><td>Conv Rec</td><td></td><td>UT↑</td><td>DNSP↑</td><td>DNS↑</td><td>CER↓</td><td>SECS↑</td></tr><tr><td>(a)</td><td>√</td><td></td><td>4.00</td><td>2.94</td><td>3.80</td><td>1.8</td><td>0.885</td></tr><tr><td>(b)</td><td></td><td></td><td>3.60</td><td>2.40</td><td>3.77</td><td>0.1</td><td>0.641</td></tr><tr><td>(c)</td><td>√</td><td>&gt;&gt;</td><td>4.04</td><td>2.97</td><td>3.80</td><td>1.5</td><td>0.885</td></tr></table>

Table 2: Analysis of adversarial training. Diffuse and Mix indicate the use of diffusion-GAN training and sample mixing, respectively.
<table><tr><td></td><td>GAN</td><td>Diffuse Mix</td><td></td><td>UT↑</td><td>DNSP↑</td><td>DNS↑</td><td>CER↓</td><td>SECS↑</td></tr><tr><td>(a)</td><td>None</td><td>一</td><td></td><td>4.04</td><td>2.79</td><td>3.75</td><td>2.2</td><td>0.882</td></tr><tr><td>(b)</td><td>Proposed</td><td></td><td></td><td>4.01</td><td>2.87</td><td>3.79</td><td>1.9</td><td>0.883</td></tr><tr><td>(c)</td><td>Proposed</td><td>√</td><td></td><td>4.04</td><td>2.86</td><td>3.78</td><td>1.5</td><td>0.882</td></tr><tr><td>(d)</td><td>Proposed</td><td></td><td>√</td><td>3.93</td><td>2.90</td><td>3.80</td><td>1.9</td><td>0.882</td></tr><tr><td>(e)</td><td>Proposed</td><td>√</td><td>√</td><td>4.04</td><td>2.97</td><td>3.80</td><td>1.5</td><td>0.885</td></tr><tr><td>(f)</td><td>WD [55]</td><td>一</td><td>一</td><td>4.04</td><td>2.89</td><td>3.79</td><td>1.8</td><td>0.884</td></tr><tr><td>(g)</td><td>VPFD [30]</td><td>一</td><td></td><td>4.02</td><td>2.90</td><td>3.79</td><td>1.8</td><td>0.884</td></tr></table>

Analysis of conditioning augmentation. We evaluated the effectiveness of the conditioning augmentation (CondAug). In addition to the standard ablation, we evaluated an alternative approach, Direct Distill, which adds an $\ell _ { 1 }$ loss to the objective without CondAug to explicitly align the student content representation $c _ { \phi }$ with the teacher representation c<sub>θ</sub>. Table 3 presents the results. CondAug (b) improved all objective metrics over the w/o CondAug configuration (a). By contrast, Direct Distill (c) did not provide comparable improvements and instead degraded DNSP and CER. Although direct feature alignment enforced similarity to the teacher representation, it appeared to overly constrain the student content encoder. These findings suggest that CondAug offers more effective implicit regularization than explicit feature-level distillation.

Table 3: Analysis ofconditioning augmentation (CondAug). Direct Distill adds an explicit $\ell _ { 1 }$ loss between the student and teacher content representations to the w/o CondAug objective.
<table><tr><td></td><td>UT↑</td><td>DNSP↑</td><td>DNS↑</td><td>CER↓</td><td>SECS↑</td></tr><tr><td>(a) w/o CondAug</td><td>4.04</td><td>2.97</td><td>3.80</td><td>1.5</td><td>0.885</td></tr><tr><td>(b) w/ CondAug</td><td>4.05</td><td>2.99</td><td>3.81</td><td>1.2</td><td>0.887</td></tr><tr><td>(c) Direct Distill</td><td>4.05</td><td>2.94</td><td>3.80</td><td>1.9</td><td>0.884</td></tr></table>

## 4.3. Comparison with previous models

We compared MeanVoiceFlow2 (MVF2) with previous models to assess its relative performance. Specifically, we compared it with MeanVoiceFlow (MVF; the teacher model) and Faster-VoiceGrad (FVG2) [34], which also jointly distills the conversion module and content encoder but adopts a diffusion-model backbone [19] and relies on a pretrained neural vocoder for stable adversarial training. We included ground-truth (GT) speech and DiffVC (30 iterations) [21] as anchor samples. For a comprehensive evaluation, we conducted MOS tests on 90 speaker– sentence pairs per model. Naturalness was rated on a five-point scale (nMOS: 1 = bad, 2 = poor, 3 = fair, 4 = good, and 5 = excellent). Speaker similarity was rated on a four-point scale (sMOS: 1 = different (sure), 2 = different (not sure), 3 = same (not sure), and 4 = same (sure)). The tests were conducted online with 12 participants, with over 1,500 ratings collected for each test. In addition, we measured the inference speed of the main conversion flow and content encoder using a real-time factor (RTF) on a single NVIDIA GeForce RTX 4090 GPU. Table 4 summarizes the results. MVF2 achieved a higher perceptual quality (nMOS, UT, DNSP, DNS) than MVF, while maintaining comparable intelligibility (CER) and speaker similarity (sMOS, SECS), and significantly reducing the real-time factor (RTF) by approximately 9×. Compared to FVG2, MVF2 achieved better or comparable performance across all metrics, with significant improvements in nMOS and DNSP and without relying on a pretrained neural vocoder for training.

Table 4: Comparison with previous models in terms of subjective metrics (nMOS and sMOS with 95% confidence intervals), objective metrics, and RTF. <sup>∗</sup> indicates a statistically significant differencefrom MVF2 on the Mann–Whitney U test $( p < 0 . 0 5 ) .$
<table><tr><td>nMOS↑</td><td>sMOS↑</td><td>UT↑</td><td>DNSP↑</td><td>DNS↑</td><td>CER↓ SECS↑</td></tr><tr><td>(a) GT 4.26±.09*</td><td>3.64±.06*</td><td>4.15</td><td>2.89 3.75</td><td>0.1</td><td>0.940 =</td></tr><tr><td>(b) DiffVC 3.43±.11</td><td>2.24±.10* 3.76</td><td>2.64</td><td>3.75</td><td>5.4</td><td>0.880 0.19</td></tr><tr><td>(c) MVF 3.76±.09</td><td>2.74±.11</td><td>3.98</td><td>2.85 3.78</td><td>1.2</td><td>0.886 0.0072</td></tr><tr><td>(d) MVF2 3.93±.10</td><td>2.70±.11</td><td>4.05 2.99</td><td>3.81</td><td>1.2</td><td>0.887 0.00084</td></tr><tr><td>(e) FVG2</td><td>3.72±.10* 2.63±.10</td><td>4.03</td><td>2.79 3.82</td><td>1.2</td><td>0.890 0.00084</td></tr></table>

## 4.4. Evaluation on LibriTTS

To assess whether our findings are robust across datasets, we evaluated MVF (teacher) and MVF2 (student) on LibriTTS [43]. As listed in Table 5, MVF2 achieved better or comparable performance across all metrics, with significant improvements in UT and DNSP while reducing the real-time factor (RTF) by approximately 9× compared to MVF. These results demonstrate that the proposed distillation framework was effective and computationally efficient for various datasets.

Table 5: Comparison on LibriTTS.
<table><tr><td></td><td>UT↑</td><td>DNSP↑</td><td>DNS↑</td><td>CER↓</td><td>SECS↑</td><td>RTF↓</td></tr><tr><td>(a) MVF</td><td>3.93</td><td>3.01</td><td>3.70</td><td>1.1</td><td>0.879</td><td>0.0089</td></tr><tr><td>(b) MVF2</td><td>4.05</td><td>3.10</td><td>3.71</td><td>1.1</td><td>0.880</td><td>0.0010</td></tr></table>

## 5. Conclusion

We presented MeanVoiceFlow2, a one-step zero-shot VC model that jointly optimizes a flow-based conversion module and an efficient content encoder. This framework integrates conversion distillation with reconstruction, diffusion-GAN training with sample mixing, and conditioning augmentation. Experiments showed that MeanVoiceFlow2 achieved higher perceptual quality than MeanVoiceFlow while maintaining comparable speaker similarity and reducing the inference time by approximately 9×. Future work will include extending the framework to practical applications such as accent and real-time VC.

## 6. Generative AI Use Disclosure

The authors used generative AI tools to edit and polish the language of the manuscript. The authors have verified the final manuscript and are responsible for its content.

## 7. References

[1] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” in ICLR, 2014.

[2] D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic backpropagation and approximate inference in deep generative models,” in ICML, 2014.

[3] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in NIPS, 2014.

[4] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in NIPS, 2017.

[5] L. Dinh, D. Krueger, and Y. Bengio, “NICE: Non-linear indepen dent components estimation,” in ICLR Workshop, 2015.

[6] C.-C. Hsu, H.-T. Hwang, Y.-C. Wu, Y. Tsao, and H.-M. Wang, “Voice conversion from unaligned corpora using variational autoencoding Wasserstein generative adversarial networks,” in Interspeech, 2017.

[7] H. Kameoka, T. Kaneko, K. Tanaka, and N. Hojo, “ACVAE-VC: Non-parallel voice conversion with auxiliary classifier variational autoencoder,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 27, no. 9, pp. 1432–1443, 2019.

[8] T. Kaneko and H. Kameoka, “CycleGAN-VC: Non-parallel voice conversion using cycle-consistent adversarial networks,” in EU-SIPCO, 2018.

[9] H. Kameoka, T. Kaneko, K. Tanaka, and N. Hojo, “StarGAN-VC: Non-parallel many-to-many voice conversion using star generative adversarial networks,” in SLT, 2018.

[10] K. Qian, Y. Zhang, S. Chang, X. Yang, and M. Hasegawa-Johnson, “AutoVC: Zero-shot voice style transfer with only autoencoder loss,” in ICML, 2019.

[11] Y.-H. Chen, D.-Y. Wu, T.-H. Wu, and H.-y. Lee, “AGAIN-VC: A one-shot voice conversion using activation guidance and adaptive instance normalization,” in ICASSP, 2021.

[12] J. Serra, S. Pascual, and C. S. Perales, “Blow: A single-scale hy-\` perconditioned flow for non-parallel raw-audio voice conversion,” in NeurIPS, 2019.

[13] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in ICML, 2015.

[14] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” in NeurIPS, 2019.

[15] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[16] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in ICLR, 2023.

[17] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in ICLR, 2023.

[18] M. S. Albergo and E. Vanden-Eijnden, “Building normalizing flows with stochastic interpolants,” in ICLR, 2023.

[19] H. Kameoka, T. Kaneko, K. Tanaka, N. Hojo, and S. Seki, “Voice-Grad: Non-parallel any-to-many voice conversion with annealed Langevin dynamics,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 32, pp. 2213–2226, 2024.

[20] S. Liu, Y. Cao, D. Su, and H. Meng, “DiffSVC: A diffusion probabilistic model for singing voice conversion,” in ASRU, 2021.

[21] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, M. Kudinov, and J. Wei, “Diffusion-based voice conversion with fast maximum likelihood sampling scheme,” in ICLR, 2022.

[22] H.-Y. Choi, S.-H. Lee, and S.-W. Lee, “Diff-HierVC: Diffusionbased hierarchical voice conversion with robust pitch generation and masked prior for zero-shot speaker adaptation,” in Interspeech, 2023.

[23] ——, “DDDM-VC: Decoupled denoising diffusion models with disentangled representation and prior mixup for verified robust voice conversion,” in AAAI, 2024.

[24] J. Yao, Y. Yuguang, Y. Pan, Z. Ning, J. Ye, H. Zhou, and L. Xie, “StableVC: Style controllable zero-shot voice conversion with conditional flow matching,” in AAAI, 2025.

[25] H.-Y. Choi and J. Park, “VoicePrompter: Robust zero-shot voice conversion with voice prompt and conditional flow matching,” in ICASSP, 2025.

[26] J. Zuo, S. Ji, M. Fang, Z. Jiang, X. Cheng, Q. Yang, W. Liu, G. Zhang, Z. Tu, Y. Guo, and Z. Zhao, “Enhancing expressive voice conversion with discrete pitch-conditioned flow matching model,” in ICASSP, 2025.

[27] P. Ren, W. Guan, K. Wang, P. Chen, Q. Hong, and L. Li, “ReFlow-VC: Zero-shot voice conversion based on rectified flow and speaker feature optimization,” in Interspeech, 2025.

[28] H. Kameoka, T. Kaneko, K. Tanaka, and Y. Kondo, “LatentVoice-Grad: Nonparallel voice conversion with latent diffusion/flowmatching models,” IEEE Trans. Audio Speech Lang. Process., vol. 33, pp. 4071–4084, 2025.

[29] T. Kaneko, H. Kameoka, K. Tanaka, and Y. Kondo, “FastVoice-Grad: One-step diffusion-based voice conversion with adversarial conditional diffusion distillation,” in Interspeech, 2024.

[30] ——, “Vocoder-projected feature discriminator,” in Interspeech, 2025.

[31] ——, “MeanVoiceFlow: One-step nonparallel voice conversion with mean flows,” in ICASSP, 2026.

[32] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He, “Mean flows for one-step generative modeling,” in NeurIPS, 2025.

[33] Z. Wang, H. Zheng, P. He, W. Chen, and M. Zhou, “Diffusion-GAN: Training GANs with diffusion,” in ICLR, 2023.

[34] T. Kaneko, H. Kameoka, K. Tanaka, and Y. Kondo, “FasterVoice-Grad: Faster one-step diffusion-based voice conversion with adversarial diffusion conversion distillation,” in Interspeech, 2025.

[35] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis,” in NeurIPS, 2020.

[36] Y. Jia, Y. Zhang, R. J. Weiss, Q. Wang, J. Shen, F. Ren, Z. Chen, P. Nguyen, R. Pang, I. L. Moreno, and Y. Wu, “Transfer learning from speaker verification to multispeaker text-to-speech synthesis,” in NeurIPS, 2018.

[37] S. Liu, Y. Cao, D. Wang, X. Wu, X. Liu, and H. Meng, “Any-to-many voice conversion with location-relative sequenceto-sequence modeling,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 29, pp. 1717–1728, 2021.

[38] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel, D. Podell, T. Dockhorn, Z. English, K. Lacey, A. Goodwin, Y. Marek, and R. Rombach, “Scaling rectified flow transformers for high-resolution image synthesis,” in ICML, 2024.

[39] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented transformer for speech recognition,” in Interspeech, 2020.

[40] Z. Geng, A. Pokle, W. Luo, J. Lin, and J. Z. Kolter, “Consistency models made easy,” in ICLR, 2025.

[41] X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. P. Smolley, “Least squares generative adversarial networks,” in ICCV, 2017.

[42] J. Yamagishi, C. Veaux, and K. MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.

[43] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “LibriTTS: A corpus derived from LibriSpeech for text-to-speech,” in Interspeech, 2019.

[44] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[45] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language modeling with gated convolutional networks,” in ICML, 2017.

[46] T. Salimans and D. P. Kingma, “Weight normalization: A simple reparameterization to accelerate training of deep neural networks,” in NIPS, 2016.

[47] D. Ulyanov, A. Vedaldi, and V. S. Lempitsky, “Instance normalization: The missing ingredient for fast stylization,” arXiv preprint arXiv:1607.08022, 2016.

[48] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in ICLR, 2015.

[49] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “UTMOS: UTokyo-SaruLab system for Voice-MOS Challenge 2022,” in Interspeech, 2022.

[50] F. Cumlin, X. Liang, V. Ungureanu, C. K. A. Reddy, C. Schuldt,¨ and S. Chatterjee, “DNSMOS Pro: A reduced-size DNN for probabilistic MOS of speech,” in Interspeech, 2024.

[51] W.-C. Huang, E. Cooper, Y. Tsao, H.-M. Wang, T. Toda, and J. Yamagishi, “The VoiceMOS Challenge 2022,” in Interspeech, 2022.

[52] C. K. A. Reddy, V. Gopal, and R. Cutler, “DNSMOS: A nonintrusive perceptual objective speech quality metric to evaluate noise suppressors,” in ICASSP, 2021.

[53] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in ICML, 2023.

[54] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, and F. Wei, “WavLM: Largescale self-supervised pre-training for full stack speech processing,” IEEE J. Sel. Top. Signal Process., vol. 16, no. 6, pp. 1505– 1518, 2022.

[55] S.-g. Lee, W. Ping, B. Ginsburg, B. Catanzaro, and S. Yoon, “BigVGAN: A universal neural vocoder with large-scale train ing,” in ICLR, 2023.