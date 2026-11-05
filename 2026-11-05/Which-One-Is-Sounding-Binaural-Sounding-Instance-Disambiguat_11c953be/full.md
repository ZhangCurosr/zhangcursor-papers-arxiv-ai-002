# Which One Is Sounding? Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation

Haoze Ma<sup>1</sup>, Yidi Li<sup>1,2∗</sup>, Yu Ma<sup>1</sup>, Pengyuan Zhao<sup>1</sup>, Yi Shi<sup>1</sup>, Bin Ren<sup>3</sup>, Yanwen Guo<sup>1</sup>, Dimitrios Kanoulas<sup>2</sup>

<sup>1</sup>Taiyuan University of Technology, China <sup>2</sup>University College London, United Kingdom <sup>3</sup>EPFL, Switzerland

## Abstract

Existing audio-visual segmentation methods mainly predict sound-related regions or category masks and often fail to identify the sounding instance when multiple same-class candidates coexist. Recent audio-visual instance segmentation methods address instance-level prediction but rely largely on motion and cross-frame association, which may become unreliable under sparse observations or inconspicuous motion. To this end, we introduce binaural audio, whose interaural diferences encode source-direction information, to provide spatial constraints for distinguishing the sounding instance among same-class visual candidates. We propose Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation (BiSID-AVS), which progressively refines category-level candidate responses into a mask of the true sounding instance. BiSID-AVS first grounds category-consistent candidate regions through audio prompting, then projects binaural cues into an image-plane spatial response map, and finally refines the true sounding instance through dual-level spatial conditioning at the feature and decoder levels. Experiments on the binaurally extended AVS-Synthetic benchmark and Habitat-AVS dataset for robotic indoor perception show that BiSID-AVS achieves the best performance on the multi-candidate subsets, improving instance-level mIoU by 35.19 and 41.21 percentage points, respectively, while substantially reducing false activation on non-sounding same-class regions.

## Introduction

Audio-visual segmentation (AVS) aims to segment visual regions corresponding to synchronized audio. Since AVSBench, existing methods have addressed single-source, multi-source, and semantic segmentation settings (Zhou et al. 2022, 2025; Yang et al. 2024b; Huang et al. 2025; Gong et al. 2025). However, their predictions generally remain sound-related regions or category masks without explicit instance identity. This limitation becomes critical when multiple same-class candidates coexist. For example, audio semantics may indicate that a phone or alarm clock is sounding but cannot determine which particular instance produces the sound, causing a model to activate several same-class objects simultaneously. Recent robustness studies also suggest that some AVS models may rely heavily on visual saliency without suficiently using audio context to identify the actual source (Li et al. 2026). Such category-level predictions are inadequate for robot interaction, source approaching, and object grasping, which require localization and segmentation of the specific sounding instance.

![](images/4d0e9757d111b41c7be9f2af20b0e1563da806e064b28db004dcd8817b054422.jpg)  
Figure 1: Task motivation of BiSID-AVS. AVS may activate multiple same-class candidates, while existing AVIS mainly relies on temporal context. BiSID-AVS instead uses binaural spatial cues to identify the true sounding instance.

Audio-visual instance segmentation (AVIS) further identifies, segments, and tracks individual sounding objects (Guo et al. 2025; Zhu et al. 2026). Both AVIS and this work address the question of which instance is sounding. Existing AVIS methods, however, mainly derive instance-discriminative evidence from continuous videos through frame-wise source localization, long-range audio-visual dependencies, and crossframe instance association. These temporal cues may become unavailable or unreliable when only a single visual observation or sparse observations are provided, or when the sounding object exhibits little visible motion.

Binaural audio provides spatial evidence complementary to video temporal information. Interaural time and level differences, together with direction-dependent spectral cues, encode the source location relative to the observer (Grothe, Pecka, and McAlpine 2010). Recent work further exploits the semantic and spatial information in binaural audio for visual sounding-object localization (Li, Zhao, and Yuan 2024), suggesting its potential as an instance-discriminative cue. Therefore, this work asks: Without reliable temporal cues, can a single visual observation and synchronized binaural audio identify the true sounding instance among multiple same-class candidates? Figure 1 illustrates the task motivation. AVS predicts sound-related regions or category masks, video-based AVIS mainly exploits temporal context to preserve instance identity, whereas our setting uses binaural spatial cues to disambiguate same-class candidates.

![](images/485b6f380491df2dacf739b2fce913dad957ea1696952abb5ee83817b9acbe10.jpg)  
Figure 2: Overall comparison between SAMA-AVS (Liu et al. 2024a) and BiSID-AVS. SAMA-AVS uses monaural audio prompting to model category-level audio-visual correspondence and may activate same-class candidates. BiSID-AVS models binaural spatial evidence and progressively refines category-consistent candidates into the mask of the true sounding instance.

A straightforward solution is to independently estimate a source direction or spatial response map, generate visual proposals, and select the proposal with the highest spatial consistency. However, such decision-level fusion introduces spatial evidence only after mask generation. It therefore cannot guide candidate representation or mask formation and may remain unreliable when spatial responses are coarse, same-class objects are adjacent, or a visual proposal contains multiple instances.

To address this problem, we propose BiSID-AVS. Existing audio-prompting methods such as SAMA-AVS (Liu et al. 2024a) establish category-level audio-visual correspondence but lack explicit spatial evidence for distinguishing sameclass instances. BiSID-AVS instead jointly models categoryconsistent candidates and binaural spatial evidence within a progressive instance-disambiguation process. It first grounds visual candidates associated with the sound category, then projects interaural cues into an image-plane spatial response, and finally incorporates this response into candidate refinement and mask prediction. This progressive process enables spatial evidence to participate directly in sounding-instance mask formation, rather than selecting among independently generated visual masks through post-hoc matching. Figure 2 illustrates the transition from semantic audio-visual correspondence to binaural sounding-instance disambiguation.

Our main contributions are summarized as follows:

• To our knowledge, this is the first work to introduce binaural spatial cues into direct pixel-level audio-visual sounding-instance segmentation, enabling identification of the true sounding instance among same-class candidates when temporal cues are unavailable or unreliable.

• We propose BiSID-AVS, a progressive framework that integrates candidate grounding, binaural spatial response modeling, and sounding-instance refinement. It directly incorporates spatial evidence into mask formation instead of relying on post-hoc matching between source localization and visual segmentation.

• We construct Binaural AVS-Synthetic and Habitat-AVS for broad-category benchmarking and geometryconsistent robotic-view disambiguation. BiSID-AVS improves mIoU<sub>M</sub> by 35.19 and 41.21 percentage points on their multi-candidate subsets, respectively, while substantially reducing same-class false activation.

## Related work

Audio-visual segmentation. Audio-visual segmentation localizes sound-related image regions from synchronized audio. AVSBench established pixel-level protocols for singlesource, multi-source, and semantic settings (Zhou et al. 2022, 2025). Subsequent work advances cross-modal interaction through Transformer-based designs, query mechanisms, and multi-order bilateral relations (Ling et al. 2024; Gao et al. 2024; Yang et al. 2024b); strengthens visual representations using vision-centric or state-space designs and foundationmodel adaptation (Huang et al. 2025; Kirillov et al. 2023; Liu et al. 2024a; Gong et al. 2025); and extends AVS to openvocabulary and weakly supervised settings (Guo et al. 2024; Mo and Raj 2023). Among SAM-based approaches, SAMA-AVS uses audio prompting to ground category-consistent visual regions (Liu et al. 2024a). However, monaural semantic prompting provides no explicit spatial evidence for distinguishing instances of the same class. Overall, existing AVS methods improve audio-visual correspondence but primarily produce sound-related region or category masks without explicit instance identity. They therefore do not directly determine which candidate is sounding when multiple same-class objects coexist in complex visual scenes.

Video-based audio-visual instance segmentation. Audio-visual instance segmentation further requires identifying, segmenting, and tracking specific sounding targets. AVISM combines frame-wise source localization, instancecontext extraction, and video-level matching to preserve instance identity, while SeaVIS studies online association and causal prediction in continuous video streams (Guo et al. 2025; Zhu et al. 2026). Existing AVIS methods therefore infer instance identity mainly from temporal context, framewise source evidence, and cross-frame instance association. In contrast, this work investigates whether binaural spatial cues can support same-class instance disambiguation when only a single or sparse visual observation is available.

![](images/bcba4e6ef913453d66f344d71b3a3d8b44d931620fa4bf5cae7920e4eb129540.jpg)  
Figure 3: Overall framework of BiSID-AVS. The model progressively performs (A) category-consistent candidate grounding, (B) binaural-guided image-plane spatial response modeling, and (C) sounding-instance mask refinement. Spatial evidence conditions both candidate-feature formation and mask decoding to refine category-consistent regions into the true sounding-instance mask.

Binaural spatial audio for embodied audio-visual perception. Binaural audio encodes source location through interaural time and level diferences and direction-dependent spectral cues (Grothe, Pecka, and McAlpine 2010). Methods exploit audio-visual correspondence and binaural cues for spatial representation learning and sounding-object localization (Gao and Grauman 2019; Morgado, Li, and Vasconcelos 2020; Chen, Qian, and Owens 2023; Li, Zhao, and Yuan 2024). SoundSpaces and SoundSpaces 2.0 support binaural rendering for embodied navigation, source localization, and acoustic reasoning (Chen et al. 2020, 2022). These methods predict source directions, navigation actions, or threedimensional scene representations rather than image-plane sounding-instance masks. Closest to our setting, 3D audiovisual segmentation lifts 2D AVS masks into a 3D Gaussian representation and refines spatial clusters with binaural audio (Sokolov, Bhosale, and Zhu 2024). BiSID-AVS instead incorporates binaural spatial evidence directly into two-dimensional mask formation, avoiding 2D-to-3D lifting and subsequent spatial-cluster refinement.

## Method

BiSID-AVS formulates sounding-instance segmentation as a progressive same-class disambiguation process. Given a visual observation $I _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ and synchronized binaural audio $A _ { t } ^ { \mathrm { b i } } = ( x _ { t } ^ { \mathrm { L } } , x _ { t } ^ { \mathrm { R } } )$ , the model predicts a soundinginstance mask $\hat { M } _ { t } ^ { \mathrm { i n s } } \in [ 0 , 1 ] ^ { H \times W }$ that covers the true sounding instance among same-class candidates. As shown in Figure 3, it first uses audio semantics to ground categoryconsistent candidate regions, then maps binaural spatial cues into a dense image-plane response, and finally incorporates this response into candidate refinement and mask decoding. The candidate-grounding and spatial-response pathways are specialized but jointly contribute to sounding-instance mask formation, avoiding post-hoc matching between independently estimated source locations and visual masks.

## Audio-Prompted Candidate Region Grounding

This stage produces category-consistent candidate features using the SAM ViT-H image encoder (Kirillov et al. 2023) and a frozen VGGish encoder (Hershey et al. 2017). We average the binaural channels and extract an audio-semantic feature from the resulting mel spectrogram:

$$
\boldsymbol { x } _ { t } ^ { \mathrm { m o n o } } = \frac { \boldsymbol { x } _ { t } ^ { \mathrm { L } } + \boldsymbol { x } _ { t } ^ { \mathrm { R } } } { 2 } , \qquad \boldsymbol { F } _ { t } ^ { \mathrm { a - s e m } } = E _ { \mathrm { v g g } } ( \mathcal { M } ( \boldsymbol { x } _ { t } ^ { \mathrm { m o n o } } ) ) .\tag{1}
$$

At SAM layer l, a lightweight mapper $g _ { l }$ converts the audiosemantic feature into a layer-wise prompt $p _ { t , l } = g _ { l } ( F _ { t } ^ { \mathrm { a - s e m } } )$ . The prompted branch broadcasts and adds $p _ { t , l }$ to the visual tokens before the Transformer block, whereas the unprompted branch processes the visual tokens directly:

$$
V _ { t , l } ^ { \mathrm { v - c a n d } } = B _ { l } \left( V _ { t , l - 1 } ^ { \mathrm { v - c a n d } } + p _ { t , l } \right) , \qquad V _ { t , l } ^ { \mathrm { v } } = B _ { l } ( V _ { t , l - 1 } ^ { \mathrm { v } } ) ,\tag{2}
$$

where $B _ { l }$ denotes the l-th SAM Transformer block. After the final block, the prompted and unprompted token sequences are reshaped and projected by the SAM encoder neck to obtain the audio-prompted candidate feature $F _ { t }$ v-cand and the unprompted visual feature $F _ { t } ^ { \mathrm { v } }$ . The former represents audio-conditioned candidate regions, while the latter preserves category-agnostic spatial structure for subsequent spatial matching. An auxiliary head predicts the candidateregion map $\hat { M } _ { t } ^ { \mathrm { c a n d } }$ , used only for auxiliary training.

## Binaural-Guided Visual Spatial Response Modeling

The second stage estimates the spatial consistency between the binaural source and each image location. We transform the left and right waveforms into cochleagrams that preserve interaural diferences (Slaney 1998), and stack them as the spatial-audio input:

$$
\boldsymbol { C } _ { t } ^ { \mathrm { L } } = \mathcal { C } ( \boldsymbol { x } _ { t } ^ { \mathrm { L } } ) , \quad \boldsymbol { C } _ { t } ^ { \mathrm { R } } = \mathcal { C } ( \boldsymbol { x } _ { t } ^ { \mathrm { R } } ) , \quad \boldsymbol { A } _ { t } ^ { \mathrm { s p l } } = \mathrm { S t a c k } ( \boldsymbol { C } _ { t } ^ { \mathrm { L } } , \boldsymbol { C } _ { t } ^ { \mathrm { R } } ) .\tag{3}
$$

A lightweight convolutional encoder with temporal downsampling and global pooling extracts a spatial feature, which is projected into the visual space as a binaural spatial query:

$$
F _ { t } ^ { \mathrm { s p l } } = E _ { \mathrm { s p l } } ( A _ { t } ^ { \mathrm { s p l } } ) \in \mathbb { R } ^ { D _ { \mathrm { s } } } , \qquad Q _ { t } ^ { \mathrm { s p l } } = W _ { \mathrm { s p l } } F _ { t } ^ { \mathrm { s p l } } \in \mathbb { R } ^ { D _ { \mathrm { v } } } .\tag{4}
$$

The query encodes receiver-relative source direction rather than a discrete image coordinate. To align it with image locations without interference from category-biased activations, we use the unprompted visual feature and add the native SAM positional encoding as an explicit image-plane reference:

$$
\widetilde { F } _ { t } ^ { \mathrm { v } } = \mathrm { s g } ( F _ { t } ^ { \mathrm { v } } ) + P ^ { \mathrm { E } } , \qquad P ^ { \mathrm { E } } \in \mathbb { R } ^ { h \times w \times D _ { \mathrm { v } } } ,\tag{5}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient, which prevents spatialresponse supervision from updating the visual backbone. A location-wise channel inner product then produces the spatial response map over the image plane:

$$
H _ { t } ^ { \mathrm { s p l } } ( u , v ) = \sum _ { d = 1 } ^ { D _ { \mathrm { v } } } \widetilde { F } _ { t , d } ^ { \mathrm { v } } ( u , v ) Q _ { t , d } ^ { \mathrm { s p l } } .\tag{6}
$$

The resulting spatial response map $H _ { t } ^ { \mathrm { s p l } } \in \mathbb { R } ^ { h \times w }$ assigns higher responses to locations more consistent with the binaural direction. Retaining this dense response, rather than regressing a discrete coordinate, allows uncertain spatial evidence to condition subsequent candidate refinement.

## Spatially Conditioned Sounding-Instance Refinement

The final stage resolves same-class ambiguity by conditioning both candidate features and mask decoding on the spatial response map to recover the true sounding instance.

Feature-level spatial modulation. We first convert the spatial response map $H _ { t } ^ { \mathrm { s p l } }$ into a bounded spatial gate and resize it to the candidate-feature resolution:

$$
G _ { t } ^ { \mathrm { s p l } } = \mathrm { R e s i z e } \left( \sigma ( H _ { t } ^ { \mathrm { s p l } } ) \right) , \qquad G _ { t } ^ { \mathrm { s p l } } \in \mathbb { R } ^ { h \times w } .\tag{7}
$$

The candidate feature is then modulated in a residual form:

$$
F _ { t } ^ { \mathrm { m o d } } = \mathrm { s g } \big ( F _ { t } ^ { \mathrm { v - c a n d } } \big ) \odot \Big ( 1 + \alpha G _ { t } ^ { \mathrm { s p l } } \Big ) ,\tag{8}
$$

where $G _ { t } ^ { \mathrm { s p l } }$ is broadcast along the channel dimension, and $F _ { t } ^ { \mathrm { m o d } }$ denotes the spatially modulated candidate feature. Stop-gradient preserves candidate grounding from instancelevel objectives. The learnable global scalar α is initialized to zero, making the modulation initially an identity mapping.

Decoder-level spatial prompting. Feature modulation reweights the candidate representation, while decoder-level prompting provides an explicit spatial condition for mask generation. A lightweight mapper $\psi$ converts the response map into a dense prompt:

$$
D _ { t } ^ { \mathrm { s p l } } = \psi ( H _ { t } ^ { \mathrm { s p l } } ) , \qquad D _ { t } ^ { \mathrm { s p l } } \in \mathbb { R } ^ { h _ { \mathrm { d } } \times w _ { \mathrm { d } } \times D _ { \mathrm { d } } } ,\tag{9}
$$

whose dimensions match the SAM dense-prompt interface. The SAM decoder $\mathcal { D } _ { \mathrm { S A M } }$ jointly receives the modulated feature, positional encoding, and spatial prompt:

$$
P _ { t } ^ { \mathrm { i n s } } = { \mathcal { D } } _ { \mathrm { S A M } } \left( F _ { t } ^ { \mathrm { m o d } } , P ^ { \mathrm { E } } , D _ { t } ^ { \mathrm { s p l } } \right) ,\tag{10}
$$

where $P _ { t } ^ { \mathrm { i n s } }$ denotes the sounding-instance logits, and no sparse prompt is used in mask decoding.

## Training Objectives for Instance Disambiguation

BiSID-AVS is trained end-to-end with four complementary objectives for sounding-instance prediction, candidate grounding, spatial-response learning, and same-class distractor suppression:

$$
{ \mathcal { L } } = \lambda ^ { \mathrm { { i n s } } } { \mathcal { L } } ^ { \mathrm { { i n s } } } + \lambda ^ { \mathrm { { c a n d } } } { \mathcal { L } } ^ { \mathrm { { c a n d } } } + \lambda ^ { \mathrm { { s p l } } } { \mathcal { L } } ^ { \mathrm { { s p l } } } + \lambda ^ { \mathrm { { s c } } } { \mathcal { L } } ^ { \mathrm { { s c } } } ,\tag{11}
$$

where the $\lambda$ terms balance the four objectives. Given the sounding-instance logits $P _ { t } ^ { \mathrm { i n s } }$ , candidate-region logits $P _ { t } ^ { \mathrm { c a n d } }$ , and spatial response map $H _ { t } ^ { \mathrm { s p l } }$ , their probability maps are $\hat { M } _ { t } ^ { \mathrm { i n s } } ~ = ~ \sigma ( P _ { t } ^ { \mathrm { i n s } } )$ $\hat { M } _ { t } ^ { \mathrm { c a n d } } = \sigma ( P _ { t } ^ { \mathrm { c a n d } } )$ , and $\hat { S } _ { t } ^ { \mathrm { s p l } } = \sigma ( H _ { t } ^ { \mathrm { s p l } } )$ . The instance and candidate predictions are upsampled to the input resolution before computing the corresponding mask losses.

Mask losses. The sounding-instance and candidate-region predictions are supervised by their corresponding masks:

$$
\begin{array} { r l r } & { \mathcal { L } ^ { \mathrm { i n s } } = \mathcal { L } _ { \mathrm { b c e } } ^ { \mathrm { l o g i t } } ( P _ { t } ^ { \mathrm { i n s } } , M _ { \mathrm { g t } } ^ { \mathrm { i n s } } ) + \mathcal { L } _ { \mathrm { i o u } } ( \hat { M } _ { t } ^ { \mathrm { i n s } } , M _ { \mathrm { g t } } ^ { \mathrm { i n s } } ) , } & \\ & { \mathcal { L } ^ { \mathrm { c a n d } } = \mathcal { L } _ { \mathrm { b b c e } } ^ { \mathrm { l o g i t } } ( P _ { t } ^ { \mathrm { c a n d } } , M _ { \mathrm { g t } } ^ { \mathrm { c a n d } } ) + \mathcal { L } _ { \mathrm { i o u } } ( \hat { M } _ { t } ^ { \mathrm { c a n d } } , M _ { \mathrm { g t } } ^ { \mathrm { c a n d } } ) . } & \end{array}\tag{12}
$$

Here, $\mathcal { L } _ { \mathrm { b b c e } } ^ { \mathrm { l o g i t } }$ denotes BCE on logits with the positive-class weight min $\mathrm { \bar { ( } } N _ { - } / N _ { + } , 1 0 0 )$ , where $N _ { + }$ and $N _ { - }$ are the foreground and background pixel counts, and $\mathcal { L } _ { \mathrm { i o u } }$ is one minus the soft intersection-over-union between the predicted probability map and ground-truth mask.

Spatial response map loss. We area-downsample the sounding instance mask and retain locations with positive target coverage as supervision on the spatial-response grid:

$$
\begin{array} { r } { M _ { \mathrm { g t } } ^ { \mathrm { s r c } } = \mathbb { I } \big [ \mathrm { D o w n } _ { \mathrm { a r e a } } ( M _ { \mathrm { g t } } ^ { \mathrm { i n s } } ) > 0 \big ] , \qquad M _ { \mathrm { g t } } ^ { \mathrm { n o n } } = 1 - M _ { \mathrm { g t } } ^ { \mathrm { s r c } } . } \end{array}\tag{13}
$$

This prevents small targets from disappearing at the response-map resolution. The spatial loss balances source enhancement and non-source suppression:

$$
\mathcal { L } ^ { \mathrm { s p l } } = - \frac { \left. M _ { \mathrm { g t } } ^ { \mathrm { s r c } } , \log \hat { S } _ { t } ^ { \mathrm { s p l } } \right. } { | M _ { \mathrm { g t } } ^ { \mathrm { s r c } } | + \epsilon } - \beta ^ { \mathrm { n o n } } \frac { \left. M _ { \mathrm { g t } } ^ { \mathrm { n o n } } , \log ( 1 - \hat { S } _ { t } ^ { \mathrm { s p l } } ) \right. } { | M _ { \mathrm { g t } } ^ { \mathrm { n o n } } | + \epsilon } ,\tag{14}
$$

where $\langle \cdot , \cdot \rangle$ denotes the sum of element-wise products over the spatial-response grid. The first term increases responses within the true source region, while the second suppresses responses elsewhere. Separate normalization balances the sparse source region and the dominant non-source region, with $\beta ^ { \mathrm { n o n } }$ controlling the suppression strength.

Same-class suppression loss. To explicitly penalize nonsounding same-class regions, the loss is defined as:

$$
\begin{array} { r l } & { M _ { \mathrm { g t } } ^ { \mathrm { s a m e } } = M _ { \mathrm { g t } } ^ { \mathrm { c a n d } } \odot ( 1 - M _ { \mathrm { g t } } ^ { \mathrm { i n s } } ) , } \\ & { \quad \quad \mathcal { L } ^ { \mathrm { s c } } = \frac { \sum _ { ( u , v ) \in \Omega _ { \mathrm { i m g } } } \hat { M } _ { t } ^ { \mathrm { i n s } } ( u , v ) M _ { \mathrm { g t } } ^ { \mathrm { s a m e } } ( u , v ) } { | M _ { \mathrm { g t } } ^ { \mathrm { s a m e } } | + \epsilon } , } \end{array}\tag{15}
$$

where $\Omega _ { \mathrm { i m g } }$ denotes the full-resolution image grid used for mask supervision. $\mathcal L ^ { \mathrm { s c } }$ is set to zero when $\breve { M } _ { \mathrm { g t } } ^ { \mathrm { s a m e } }$ is empty.

All losses are jointly optimized. ${ \mathcal { L } } ^ { \mathrm { c a n d } }$ supervises candidate grounding, $\bar { \mathcal { L } } ^ { \mathrm { s p l } }$ directly supervises the spatial encoder and query projection, and $\dot { \mathcal { L } } ^ { \mathrm { i n s } }$ and $\mathcal L ^ { \mathrm { s c } }$ further update the spatial-response pathway, conditioning modules, and mask decoder. Stop-gradient isolates the unprompted visual anchor and candidate-grounding pathway from unintended updates.

## Experiments

## Datasets and Experimental Setup

Datasets. To evaluate sounding-instance disambiguation under controlled and geometry-consistent conditions, we use two complementary datasets. Both provide RGB images, synchronized binaural audio, sounding-instance masks and boxes, and same-class candidate masks. The soundinginstance annotation identifies the audio-associated target, and the candidate mask covers all instances of the sound category.

Binaural AVS-Synthetic extends AVS-Synthetic (Liu et al. 2024a), a commonly used synthetic benchmark for audiovisual segmentation, with instance-level annotations and binaural audio. We verify instances in the original semantic masks using GroundingDINO (Liu et al. 2024b), select one as the sounding target, and derive its rendering parameters from image position and relative depth estimated by Depth Anything (Yang et al. 2024a). Binaural audio is synthesized using the SADIE D1 HRTF (Armstrong et al. 2018). The dataset contains 31,045 samples from 46 categories, including 14.5% multi-candidate samples.

Habitat-AVS is constructed with Habitat, SoundSpaces, and Replica (Savva et al. 2019; Chen et al. 2020, 2022; Straub et al. 2019). Habitat provides image-aligned instance annotations, while SoundSpaces renders binaural audio from the target’s three-dimensional location. The dataset contains 11,781 samples from 14 categories across four indoor scenes and ten predefined robotic viewpoints, with 45.3% containing multiple same-class candidates. Together, Binaural AVS-Synthetic provides broad-category evaluation under controlled image– audio spatial correspondence, whereas Habitat-AVS evaluates geometry-consistent disambiguation with more realistic three-dimensional acoustics and robotic viewpoints.

Evaluation metrics. We report instance-level mIoU, computed as the mean per-sample IoU, mean Sounding Instance Localization Accuracy (mSILA) adapted from Frame-level

Sound Location Accuracy (FSLA) (Guo et al. 2025), sameclass false activation $F _ { \mathrm { s a m e } } .$ , and the Same-Class Disambiguation Score (SC-DS). Given the binarized prediction $\widetilde { M } _ { i } ^ { \mathrm { i n s } } = \mathbb { I } [ \hat { M } _ { i } ^ { \mathrm { i n s } } \geq 0 . 5 ]$ , the metrics are:

$$
\begin{array} { r l } & { \mathrm { m S I L A } = \displaystyle \frac { 1 } { N | T | } \sum _ { i = 1 } ^ { N } \sum _ { \tau \in \mathcal { T } } \mathbb { I } \Big [ \mathrm { I o U } ( \widetilde { M } _ { i } ^ { \mathrm { i n s } } , M _ { \mathrm { g t } , i } ^ { \mathrm { i n s } } ) \geq \tau \Big ] ~ , } \\ & { \quad F _ { \mathrm { s a m e } } ^ { ( i ) } = \displaystyle \frac { | \widetilde { M } _ { i } ^ { \mathrm { i n s } } \cap M _ { \mathrm { g t } , i } ^ { \mathrm { s a m e } } | } { | M _ { \mathrm { g t } , i } ^ { \mathrm { s a m e } } | + \epsilon } , } \\ & { \mathrm { S C } \mathrm { - D S } ^ { ( i ) } = \mathrm { I o U } ( \widetilde { M } _ { i } ^ { \mathrm { i n s } } , M _ { \mathrm { g t } , i } ^ { \mathrm { i n s } } ) \left( 1 - F _ { \mathrm { s a m e } } ^ { ( i ) } \right) , } \end{array}\tag{16}
$$

where $\mathcal { T } = \{ 0 . 5 0 , 0 . 5 5 , \hdots , 0 . 9 5 \}$ and $M _ { \mathrm { g t } , i } ^ { \mathrm { s a m e } } = M _ { \mathrm { g t } , i } ^ { \mathrm { c a n d } } \odot$ $( 1 - M _ { \mathrm { g t } , i } ^ { \mathrm { i n s } } )$ denotes the non-sounding same-class region. $F _ { \mathrm { s a m e } }$ and SC-DS are averaged over multi-candidate samples. The subscripts S, M, and All denote the singlecandidate, multi-candidate, and full sets, respectively.

Protocols and implementation details. We use five-fold cross-validation on both datasets. Binaural AVS-Synthetic is grouped by original sample and stratified by category and candidate setting. For Habitat-AVS, the ten viewpoints form five fixed pairs, with one pair held out per fold for viewpointdisjoint evaluation; we also perform four-fold leave-onescene-out evaluation. In each fold, 10% of the training data is reserved for validation, and the checkpoint with the highest validation $\mathrm { m I o U _ { A l l } }$ is used for testing. BiSID-AVS uses SAM ViT-H initialized from the oficial checkpoint and VG-Gish pretrained on AudioSet. The VGGish and SAM image encoders are frozen, while the SAM mask decoder and all newly introduced modules are optimized. Images are resized to $5 1 \dot { 2 } \times 5 1 2 .$ , and 2-s audio clips sampled at 16 kHz are converted into cochleagrams using 66 filters spanning 50–8000 Hz. We set $\lambda ^ { \mathrm { i n s } } \stackrel {  } { = } \lambda ^ { \mathrm { c a n d } } \stackrel {  } { = } \stackrel { \sim } { 1 } . 0 , \lambda ^ { \mathrm { s p l } } \stackrel {  } { = } 0 . 5 , \breve { \lambda } ^ { \mathrm { s c } } = 0 . 1$ and $\beta ^ { \mathrm { n o n } } = 1 . 0$ . Training uses AdamW for 40 epochs with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , weight decay of 0.01, cosine decay without warm-up, and batch size 1. Each fold is trained once with a fixed random seed on one NVIDIA RTX 4090 GPU. Details are provided in Appendix A.

## Comparison with Existing Methods

Main quantitative comparison. We compare BiSID-AVS with representative AVS methods, including TPAVI (Zhou et al. 2022), AVSegFormer (Gao et al. 2024), COMBO (Yang et al. 2024b), VCT (Huang et al. 2025), and SAMA-AVS (Liu et al. 2024a), as well as the sounding-instance method AVISM (Guo et al. 2025). All baselines are retrained on identical splits using public implementations and recommended settings, with only necessary input and output adaptations. Monaural methods use the channel-averaged audio, while video-based methods receive repeated static frames. Following its original protocol, AVISM additionally uses the sound category and number of sounding instances. As shown in Table 1, BiSID-AVS improves mIoU<sub>M</sub> by 35.19 and 41.21 percentage points and $\mathrm { m S I L A _ { M } }$ by $4 7 . 9 1$ and 71.43 points on two datasets. It also achieves the lowest $F _ { \mathrm { s a m e , M } }$ and highest $\mathrm { S C - D S _ { M } }$ on both datasets, demonstrating stronger soundinginstance coverage and same-class distractor suppression. On single-candidate samples, where candidate grounding is usually suficient, spatial conditioning brings limited gains and may slightly reduce mask completeness.

<table><tr><td>Dataset</td><td>Method</td><td> $\mathrm { m I o U _ { S } ^ { \uparrow } }$ </td><td> $\mathrm { m I o U _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m I o U } _ { \mathrm { A l l } } ^ { \uparrow }$ </td><td> $\mathrm { \ m S I L A _ { S } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { A l l } ^ { \uparrow } }$ </td><td> $F _ { \mathrm { s a m e , M } } ^ { \downarrow }$ </td><td> $\mathrm { S C - D S _ { M } ^ { \uparrow } }$ </td></tr><tr><td rowspan="7">Binaural AVS-Synthetic</td><td>TPAVI</td><td>0.7838</td><td>0.2742</td><td>0.7099</td><td>0.7587</td><td>0.0985</td><td>0.6629</td><td>0.6841</td><td>0.1184</td></tr><tr><td>AVSegFormer</td><td>0.8619</td><td>0.3275</td><td>0.7844</td><td>0.8213</td><td>0.1308</td><td>0.7211</td><td>0.7160</td><td>0.1067</td></tr><tr><td>COMBO</td><td>0.7695</td><td>0.2641</td><td>0.6962</td><td>0.7397</td><td>0.0953</td><td>0.6462</td><td>0.6752</td><td>0.1037</td></tr><tr><td>VCT</td><td>0.8603</td><td>0.3482</td><td>0.7860</td><td>0.8178</td><td>0.1297</td><td>0.7183</td><td>0.7227</td><td>0.0598</td></tr><tr><td>AVISM</td><td>0.8217</td><td>0.3376</td><td>0.7515</td><td>0.7839</td><td>0.1035</td><td>0.6852</td><td>0.7065</td><td>0.0724</td></tr><tr><td>SAMA-AVS</td><td>0.8426</td><td>0.3339</td><td>0.7688</td><td>0.8093</td><td>0.1187</td><td>0.7091</td><td>0.7143</td><td>0.0916</td></tr><tr><td>BiSID-AVS (Ours)</td><td>0.8386</td><td>0.7001</td><td>0.8185</td><td>0.7942</td><td>0.6099</td><td>0.7674</td><td>0.1165</td><td>0.6544</td></tr><tr><td rowspan="7">Habitat-AVS</td><td>TPAVI</td><td>0.9257</td><td>0.4670</td><td>0.7177</td><td>0.8998</td><td>0.1250</td><td>0.5486</td><td>0.9583</td><td>0.0209</td></tr><tr><td>AVSegFormer</td><td>0.9448</td><td>0.4831</td><td>0.7356</td><td>0.9517</td><td>0.1421</td><td>0.5849</td><td>0.9780</td><td>0.0112</td></tr><tr><td>COMBO</td><td>0.8828</td><td>0.4473</td><td>0.6853</td><td>0.8197</td><td>0.1069</td><td>0.4968</td><td>0.9278</td><td>0.0341</td></tr><tr><td>VCT</td><td>0.9375</td><td>0.4714</td><td>0.7262</td><td>0.9222</td><td>0.1290</td><td>0.5625</td><td>0.9629</td><td>0.0185</td></tr><tr><td>AVISM</td><td>0.9137</td><td>0.4528</td><td>0.7049</td><td>0.8725</td><td>0.1172</td><td>0.5303</td><td>0.9308</td><td>0.0293</td></tr><tr><td>SAMA-AVS</td><td>0.9368</td><td>0.4811</td><td>0.7303</td><td>0.9544</td><td>0.1401</td><td>0.5855</td><td>0.9762</td><td>0.0123</td></tr><tr><td>BiSID-AVS (Ours)</td><td>0.9321</td><td>0.8952</td><td>0.9154</td><td>0.9263</td><td>0.8564</td><td>0.8946</td><td>0.0356</td><td>0.8811</td></tr><tr><td>Habitat-AVS</td><td>SAMA-AVS</td><td>0.9219</td><td>0.4537</td><td>0.7098</td><td>0.9129</td><td>0.1042</td><td>0.5465</td><td>0.9413</td><td>0.0249</td></tr><tr><td>(scene-disjoint)</td><td>BiSID-AVS (Ours)</td><td>0.9147</td><td>0.8543</td><td>0.8873</td><td>0.8937</td><td>0.8216</td><td>0.8610</td><td>0.0437</td><td>0.8473</td></tr></table>

Table 1: Comparison of BiSID-AVS with representative audio-visual segmentation and audio-visual instance segmentation methods. The final block reports four-fold leave-one-scene-out evaluation on Habitat-AVS. S, M, and All denote the singlecandidate subset, multi-same-class-candidate subset, and all samples, respectively.

<table><tr><td>Method</td><td> $\mathrm { m I o U _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { M } ^ { \uparrow } }$ </td><td> $F _ { \mathrm { s a m e , M } } ^ { \downarrow }$ </td><td> $\mathrm { S C - D S _ { M } ^ { \uparrow } }$ </td></tr><tr><td>SSL-HM + VP</td><td>0.2956</td><td>0.2383</td><td>0.1429</td><td>0.2849</td></tr><tr><td>Cand. + SSL-HM + VP</td><td>0.7599</td><td>0.6731</td><td>0.1554</td><td>0.7419</td></tr><tr><td>GT Cand. + SSL-HM</td><td>0.9326</td><td>0.8928</td><td>0.1695</td><td>0.8617</td></tr><tr><td>BiSID-AVS (Ours)</td><td>0.8952</td><td>0.8564</td><td>0.0356</td><td>0.8811</td></tr></table>

Table 2: Decision-level fusion baselines and oracle analysis on the Habitat-AVS multi-candidate subset. GT Cand. denotes the ground-truth candidate mask.

Scene-disjoint generalization. Under the leave-onescene-out protocol, both methods show performance degradation due to unseen visual layouts and acoustic environments. BiSID-AVS substantially outperforms SAMA-AVS on the multi-candidate and full sets and achieves markedly better same-class disambiguation, while remaining slightly lower on the single-candidate metrics. These results demonstrate stronger generalization to unseen scenes in the challenging same-class disambiguation setting.

## Analysis of Binaural Spatial Disambiguation

Comparison with decision-level fusion. We construct three output-stage baselines using the upsampled spatial response as a Sound Source Localization Heatmap (SSL-HM) and Visual Proposals (VP) from frozen SAM ViT-H. SSL-HM + VP selects the proposal with the highest mean response. Cand. Map + SSL-HM + VP weights the heatmap with the predicted candidate-region probability map, whereas GT Cand. + SSL-HM restricts it using the ground-truth candidate mask. All baselines incorporate spatial evidence only through post-hoc output-stage fusion. As shown in Table 2, candidate-map weighting substantially improves $\mathrm { m I o U } _ { \mathrm { M } }$ but increases $F _ { \mathrm { s a m e , M } }$ , indicating stronger activation of same-class distractors. The GT-candidate baseline achieves the highest mIoU $\mathrm { J _ { M } }$ and mSILA<sub>M</sub>, yet retains an $F _ { \mathrm { s a m e , M } }$ of0.1695. Without GT information, BiSID-AVS obtains the lowest $F _ { \mathrm { s a m e , M } }$ and highest SC-DS , showing that spatial conditioning during refinement and decoding suppresses distractors more efectively than post-hoc selection.

<table><tr><td>Audio condition</td><td> $\mathrm { m I o U _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { M } ^ { \uparrow } }$ </td><td> $F _ { \mathrm { s a m e , M } } ^ { \downarrow }$ </td><td> $\mathrm { S C - D S _ { M } ^ { \uparrow } }$ </td></tr><tr><td>Original</td><td>0.8952</td><td>0.8564</td><td>0.0356</td><td>0.8811</td></tr><tr><td>Swapped channels</td><td>0.1737</td><td>0.1297</td><td>0.4483</td><td>0.1685</td></tr><tr><td>Mismatched audio</td><td>0.1141</td><td>0.0749</td><td>0.6437</td><td>0.1054</td></tr></table>

Table 3: Efect of interaural spatial cues on the Habitat-AVS multi-candidate subset.

Efect of interaural spatial cues. To verify that BiSID-AVS exploits binaural spatial evidence for same-class disambiguation, we perturb the audio input while keeping the image fixed. Channel swapping reverses the left–right structure, whereas mismatched audio uses a same-scene, samecategory sample from another source location. As shown in Table 3, both perturbations markedly reduce mIoU and $\mathrm { m S I L A _ { M } }$ and increase $F _ { \mathrm { s a m e , M } }$ , with mismatched audio causing the largest degradation. This confirms that BiSID-AVS relies on interaural structure and audio–visual spatial correspondence, rather than sound-category semantics alone.

## Ablation Studies

Architecture ablation. Table 4 evaluates the two functional pathways and the two components of dual-level spatial conditioning. Cfg. A retains only Candidate Grounding (CG) and performs poorly on same-class disambiguation, showing that category-consistent candidates alone are insuficient.

<table><tr><td>Cfg.</td><td>CG SR</td><td>FM DP</td><td></td><td> $\mathrm { m I o U _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { M } ^ { \uparrow } }$ </td><td> $F _ { \mathrm { s a m e , M } } ^ { \downarrow }$ </td><td> $\mathrm { S C - D S _ { M } ^ { \uparrow } }$ </td></tr><tr><td>A</td><td>√</td><td>一 一</td><td></td><td>0.4799</td><td>0.1394</td><td>0.9762</td><td>0.0123</td></tr><tr><td>B</td><td>一</td><td>√ 一</td><td>√</td><td>0.8427</td><td>0.8132</td><td>0.0461</td><td>0.8365</td></tr><tr><td>C</td><td>√</td><td>√ 一</td><td>一</td><td>0.8391</td><td>0.8091</td><td>0.1273</td><td>0.8227</td></tr><tr><td>D</td><td>√</td><td>√ √</td><td></td><td>0.8532</td><td>0.8197</td><td>0.0842</td><td>0.8415</td></tr><tr><td>E</td><td>√</td><td>√ 一</td><td>√</td><td>0.8676</td><td>0.8379</td><td>0.1039</td><td>0.8593</td></tr><tr><td>F</td><td>√</td><td>√ √</td><td>√</td><td>0.8952</td><td>0.8564</td><td>0.0356</td><td>0.8811</td></tr></table>

Table 4: Architecture ablation. CG: candidate grounding, SR: spatial response modeling, FM: feature-level modulation, DP: decoder-level prompting.  
![](images/5fdc37734bee4c79979b96809efd73b7c7f9f1cc1712458c61560d238c2c282d.jpg)  
Figure 4: Qualitative comparison on representative multisame-class-candidate scenes on two datasets.

Cfg. B combines Spatial Response (SR) with Decoder-level Prompting (DP), since SR does not directly produce a mask, and achieves strong localization performance without candidate grounding. Cfg. C retains CG and SR but disables both conditioning mechanisms. Its weaker results show that computing candidate and spatial features alone does not effectively integrate spatial evidence into mask prediction. Cfg. D and E add Feature-level Modulation (FM) and DP separately. FM suppresses spatially inconsistent candidate features, whereas DP directly conditions mask decoding. Compared with B, E improves target coverage through candidate grounding but may also increase activation on same-class distractors. Cfg. F combines FM and DP and achieves the best overall performance, confirming that feature modulation and decoder prompting provide complementary ways to incorporate binaural spatial evidence.

Loss function ablation. As shown in Table 5, replacing the bounding-box supervision $\mathcal { L } ^ { \mathrm { b o x } }$ with the spatial response map loss $\mathcal { L } ^ { \mathrm { s p l } }$ improves mIoU and reduces $F _ { \mathrm { s a m e , M } } .$ indicating that instance-mask-based spatial supervision is more precise than coarse box supervision. Adding $\mathcal L ^ { \mathrm { s c } }$ further improves both settings by explicitly suppressing same-class distractors. Their combination achieves the best results across all metrics, confirming the complementary efects of spatialresponse supervision and same-class suppression.

## Qualitative Results and Real-Robot Demonstration

Qualitative comparison. Figure 4 shows that SAMA-AVS and AVISM often activate same-class distractors together with the source. Decision-level fusion (Cand. Map + SSL-HM + VP) removes some false regions but remains unreliable for adjacent instances or proposals covering multiple objects. In contrast, BiSID-AVS consistently focuses on the sounding instance and better preserves its contour across both datasets, including small and closely spaced targets. These results are consistent with its lower $F _ { \mathrm { s a m e , M } }$ and higher SC-DS<sub>M</sub>.

<table><tr><td>Loss setting</td><td> $\mathrm { m I o U _ { M } ^ { \uparrow } }$ </td><td> $\mathrm { m S I L A _ { M } ^ { \uparrow } }$ </td><td> $F _ { \mathrm { s a m e , M } } ^ { \downarrow }$ </td><td> $\mathrm { S C - D S _ { M } ^ { \uparrow } }$ </td></tr><tr><td> $\mathcal { L } ^ { \mathrm { b a s e } } + \mathcal { L } ^ { \mathrm { b o x } }$ </td><td>0.8674</td><td>0.8219</td><td>0.1047</td><td>0.8425</td></tr><tr><td> $\mathcal { L } ^ { \mathrm { b a s e } } + \mathcal { L } ^ { \mathrm { s p l } }$ </td><td>0.8791</td><td>0.8372</td><td>0.0689</td><td>0.8679</td></tr><tr><td> $\mathcal { L } ^ { \mathrm { b a s e } } + \mathcal { L } ^ { \mathrm { b o x } } + \mathcal { L } ^ { \mathrm { s c } }$ </td><td>0.8846</td><td>0.8392</td><td>0.0428</td><td>0.8721</td></tr><tr><td> $\mathcal { L } ^ { \mathrm { b a s e } } + \mathcal { L } ^ { \mathrm { s p l } } + \mathcal { L } ^ { \mathrm { s c } }$ </td><td>0.8952</td><td>0.8564</td><td>0.0356</td><td>0.8811</td></tr></table>

Table 5: Loss-function ablation on the Habitat-AVS multicandidate subset.

![](images/2e5bce276eb28f8fb9314b373b13555701c5a215203396053e1cfdc8dc9058b4.jpg)  
Figure 5: Real-robot sounding-instance-guided navigation.

Real-robot demonstration. As shown in Figure 5, BiSID-AVS is deployed on a mobile robot equipped with binaural microphones and an RGB-D camera. Among three sameclass devices, it identifies the sounding target, whose masked depth is then converted into a navigation goal. This demonstrates the feasibility of using BiSID-AVS as a perceptual front end for robotic navigation toward a sounding object.

## Conclusion

This work establishes binaural sounding-instance disambiguation as a distinct setting for audio-visual segmentation and introduces BiSID-AVS to identify the true sounding instance among same-class visual candidates. By integrating candidate grounding and binaural spatial response modeling directly into mask prediction, the framework avoids post-hoc matching between independently estimated source locations and visual masks. The proposed Binaural AVS-Synthetic and Habitat-AVS benchmarks, together with instance-level disambiguation metrics, provide a basis for studying spatially grounded audio-visual segmentation under controlled and robotic-view conditions. Experiments demonstrate that BiSID-AVS substantially improves multi-candidate segmentation and suppresses same-class false activation. The current formulation assumes one active source per audio segment, and future work will focus on multi-source sounding-instance segmentation under overlapping acoustic events.

## References

Armstrong, C.; Thresh, L.; Murphy, D. T.; and Kearney, G. C. 2018. A Perceptual Evaluation of Individual and Non-Individual HRTFs: A Case Study of the SADIE II Database. Applied Sciences, 8(11): 2029.

AudioLab, University of York; and Kearney, G. C. 2024. SADIE II Database. Zenodo. https://doi.org/10.5281/zenodo.10886409. Version 2-1.

Chen, C.; Jain, U.; Schissler, C.; Gari, S. V. A.; Al-Halah, Z.; Ithapu, V. K.; Robinson, P.; and Grauman, K. 2020. SoundSpaces: Audio-Visual Navigation in 3D Environments. In European Conference on Computer Vision, 17–36.

Chen, C.; Schissler, C.; Garg, S.; Kobernik, P.; Clegg, A.; Calamia, P.; Batra, D.; Robinson, P. W.; and Grauman, K. 2022. SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning. In Advances in Neural Information Processing Systems, volume 35, 8896–8911.

Chen, Z.; Qian, S.; and Owens, A. 2023. Sound Localization from Motion: Jointly Learning Sound Direction and Camera Rotation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7897–7908.

Gao, R.; and Grauman, K. 2019. 2.5D Visual Sound. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 324–333.

Gao, S.; Chen, Z.; Chen, G.; Wang, W.; and Lu, T. 2024. AVSeg-Former: Audio-Visual Segmentation with Transformer. In Proceed ings of the AAAI Conference on Artificial Intelligence, volume 38, 12155–12163.

Gemmeke, J. F.; Ellis, D. P. W.; Freedman, D.; Jansen, A.; Lawrence, W.; Moore, R. C.; Plakal, M.; and Ritter, M. 2017. Audio Set: An Ontology and Human-Labeled Dataset for Audio Events. In IEEE International Conference on Acoustics, Speech and Signal Processing, 776–780.

Gong, S.; Zhuge, Y.; Zhang, L.; Wang, Y.; Zhang, P.; Wang, L.; and Lu, H. 2025. AVS-Mamba: Exploring Temporal and Multi-Modal Mamba for Audio-Visual Segmentation. IEEE Transactions on Multimedia, 27: 5413–5425.

Grothe, B.; Pecka, M.; and McAlpine, D. 2010. Mechanisms of Sound Localization in Mammals. Physiological Reviews, 90(3): 983–1012.

Guo, R.; Qu, L.; Niu, D.; Qi, Y.; Yue, W.; Shi, J.; Xing, B.; and Ying, X. 2024. Open-Vocabulary Audio-Visual Semantic Segmentation. In Proceedings of the 32nd ACM International Conference on Multimedia, 7533–7541.

Guo, R.; Ying, X.; Chen, Y.; Niu, D.; Li, G.; Qu, L.; Qi, Y.; Zhou, J.; Xing, B.; Yue, W.; Shi, J.; Wang, Q.; Zhang, P.; and Liang, B. 2025. Audio-Visual Instance Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13550–13560.

Hershey, S.; Chaudhuri, S.; Ellis, D. P. W.; Gemmeke, J. F.; Jansen, A.; Moore, R. C.; Plakal, M.; Platt, D.; Saurous, R. A.; Seybold, B.; Slaney, M.; Weiss, R. J.; and Wilson, K. 2017. CNN Architectures for Large-Scale Audio Classification. In IEEE International Conference on Acoustics, Speech and Signal Processing, 131–135.

Huang, S.; Ling, R.; Hui, T.; Li, H.; Zhou, X.; Zhang, S.; Liu, S.; Hong, R.; and Wang, M. 2025. Revisiting Audio-Visual Segmentation with Vision-Centric Transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8352–8361.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; Dollár, P.;

and Girshick, R. 2023. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4015– 4026.

Li, J.; Zhao, W.; Huang, Z.; Guo, Y.; and Tian, Y. 2026. Do Audio-Visual Segmentation Models Truly Segment Sounding Objects? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 6172–6180.

Li, Z.; Zhao, B.; and Yuan, Y. 2024. Cyclic Learning for Binaural Audio Generation and Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26669–26678.

Ling, Y.; Li, Y.; Gan, Z.; Zhang, J.; Chi, M.; and Wang, Y. 2024. TransAVS: End-to-End Audio-Visual Segmentation with Transformer. In IEEE International Conference on Acoustics, Speech and Signal Processing, 7845–7849.

Liu, J.; Wang, Y.; Ju, C.; Ma, C.; Zhang, Y.; and Xie, W. 2024a. Annotation-Free Audio-Visual Segmentation. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 5604–5614.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; Zhu, J.; and Zhang, L. 2024b. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. In European Conference on Computer Vision, 38–55.

Mo, S.; and Raj, B. 2023. Weakly-Supervised Audio-Visual Segmentation. In Advances in Neural Information Processing Systems, volume 36, 17208–17221.

Morgado, P.; Li, Y.; and Vasconcelos, N. 2020. Learning Representations from Audio-Visual Spatial Alignment. In Advances in Neural Information Processing Systems, volume 33, 4733–4744.

Savva, M.; Kadian, A.; Maksymets, O.; Zhao, Y.; Wijmans, E.; Jain, B.; Straub, J.; Liu, J.; Koltun, V.; Malik, J.; Parikh, D.; and Batra, D. 2019. Habitat: A Platform for Embodied AI Research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 9339–9347.

Slaney, M. 1998. Auditory Toolbox Version 2. Technical Report 1998-010, Interval Research Corporation.

Sokolov, A.; Bhosale, S.; and Zhu, X. 2024. 3D Audio-Visual Segmentation. In Audio Imagination: NeurIPS 2024 Workshop on AI-Driven Speech, Music, and Sound Generation.

Straub, J.; Whelan, T.; Ma, L.; Chen, Y.; Wijmans, E.; Green, S.; Engel, J. J.; Mur-Artal, R.; Ren, C.; Verma, S.; Clarkson, A.; Yan, M.; Budge, B.; Yan, Y.; Pan, X.; Yon, J.; Zou, Y.; Leon, K.; Carter, N.; Briales, J.; Gillingham, T.; Mueggler, E.; Pesqueira, L.; Savva, M.; Batra, D.; Strasdat, H. M.; Nardi, R. D.; Goesele, M.; Lovegrove, S.; and Newcombe, R. 2019. The Replica Dataset: A Digital Replica of Indoor Spaces. arXiv preprint arXiv:1906.05797.

Yang, L.; Kang, B.; Huang, Z.; Xu, X.; Feng, J.; and Zhao, H. 2024a. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10371–10381.

Yang, Q.; Nie, X.; Li, T.; Gao, P.; Guo, Y.; Zhen, C.; Yan, P.; and Xiang, S. 2024b. Cooperation Does Matter: Exploring Multi-Order Bilateral Relations for Audio-Visual Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27134–27143.

Zhou, J.; Shen, X.; Wang, J.; Zhang, J.; Sun, W.; Zhang, J.; Birchfield, S.; Guo, D.; Kong, L.; Wang, M.; and Zhong, Y. 2025. Audio-Visual Segmentation with Semantics. International Journal ofComputer Vision, 133(4): 1644–1664.

Zhou, J.; Wang, J.; Zhang, J.; Sun, W.; Zhang, J.; Birchfield, S.; Guo, D.; Kong, L.; Wang, M.; and Zhong, Y. 2022. Audio-Visual Segmentation. In European Conference on Computer Vision, 386– 403.

Zhu, Y.; Wang, Y.; Hong, Y.; Guo, R.; Ding, K.; Gu, X.; Fan, B.; and Xiang, S. 2026. SeaVIS: Sound-Enhanced Association for Online Audio-Visual Instance Segmentation. arXiv preprint arXiv:2603.01431.

## Appendix

## A Implementation Details of BiSID-AVS

## A.1 Input Representations

All RGB images are resized to 512 × 512. The visual backbone is SAM ViT-H (Kirillov et al. 2023), initialized from the oficial sam\_vit\_h\_4b8939.pth checkpoint. It uses a patch size of 16, a token dimension of 1280, 32 Transformer blocks, 16 attention heads, and a 256-channel output neck. The resulting visual feature map has a spatial resolution of $3 2 \times 3 2$

The audio-semantic prompting path first averages the synchronized left and right waveforms. Following the VGGish preprocessing pipeline (Hershey et al. 2017), the monaural waveform is resampled to 16 kHz and converted into a 96×64 log-Mel patch using a 25-ms analysis window, a 10-ms hop, and 64 Mel bands spanning 125–7500 Hz. A frozen VGGish encoder initialized from the vggish-10086976.pth checkpoint pretrained on AudioSet (Gemmeke et al. 2017) produces a 128-dimensional embedding. For each of the 32 SAM blocks, a block-specific linear layer maps this embedding from 128 to 512 dimensions, followed by GELU; a shared linear layer then projects it to the 1280-dimensional visual-token space. The resulting prompt is added to every visual token before the corresponding Transformer block. The unprompted visual representation is computed with the same SAM parameters but without audio-prompt injection.

For binaural spatial modeling, both waveforms are resampled to 16 kHz and cropped or zero-padded to 2 s. A 66-channel cochleagram spanning 50–8000 Hz with a sample factor of 1 is computed independently for each ear, and the two cochleagrams are stacked to form an input of size $2 \times 6 6 \times T$ . The two ears are never averaged in this path.

## A.2 Spatial Response and Conditioning Modules

Table A1 summarizes the key modules used in BiSID-AVS. The binaural spatial encoder uses three temporally strided convolutions, each followed by instance normalization and ReLU. Adaptive average pooling and a linear projection produce a 256-dimensional spatial audio query. A channel-wise inner product between this query and the stop-gradient unprompted visual feature plus SAM positional encoding gives the 32 × 32 source response. The dense prompt mapper converts the one-channel response into a 256-channel prompt for the SAM mask decoder.

Feature-level conditioning converts the spatial response into a bounded sigmoid gate and applies it to the detached audio-prompted candidate feature through $1 + \alpha G ^ { \mathrm { s p l } }$ . The scalar α is a single globally shared, unconstrained learnable parameter initialized to zero, so the module begins as an identity mapping. Decoder-level conditioning uses the mapped spatial response as the dense prompt. The sparseprompt input is an empty tensor; no point or box prompt is used.

## A.3 Optimization and Inference

VGGish and the non-prompt parameters of the SAM visual encoder are frozen. Training updates the audio-prompt mappings, spatial encoder and query projection, candidate-region auxiliary head, dense prompt mapper, spatial gate, and SAM mask decoder. We train for 40 epochs with AdamW, an initial learning rate o $2 \times 1 0 ^ { - 4 }$ , weight decay of 0.01, cosine decay to $1 0 ^ { - 7 }$ without warm-up, batch size 1, a fixed random seed of 42, and one NVIDIA RTX 4090 GPU. The loss weights are $\lambda ^ { \mathrm { i n s } } = 1 . 0 , \lambda ^ { \mathrm { c a n d } } = 1 . 0 , \lambda ^ { \mathrm { s p l } } = 0 . 5 , \lambda ^ { \mathrm { s c } } = 0 . 1$ , and $\beta ^ { \mathrm { n o n } } = 1 . 0$ . Model selection uses the validation partition of each fold, with $\mathrm { m I o U _ { A l l } }$ as the selection metric. At inference, the instance-mask logits are converted to probabilities with a sigmoid and binarized at 0.5 for evaluation.

## A.4 Loss and Ablation Details

The soft IoU loss used for the sounding-instance and candidate-region predictions is one minus the standard soft intersection-over-union and is averaged over the batch. For the candidate-region BCE, the positive-class weight is min $\left( N _ { - } / ( N _ { + } + \epsilon ) \right.$ , 100), where $N _ { + }$ and $N _ { - }$ denote the foreground and background pixel counts and ϵ prevents division by zero. The spatial response loss separately normalizes its source and non-source terms, as defined in the main paper.

For the loss ablation, ${ \mathcal { L } } ^ { \mathrm { b a s e } } = \lambda ^ { \mathrm { i n s } } { \mathcal { L } } ^ { \mathrm { i n s } } + \lambda ^ { \mathrm { c a n d } } { \mathcal { L } }$ cand collects the instance-mask and candidate-region objectives. $\mathcal { L } ^ { \mathrm { b o x } }$ denotes the alternative bounding-box spatial supervision: a binary target is set to one inside the sounding-instance box and zero elsewhere, and binary cross-entropy is applied to the spatial-response logits. The fixed loss weights are omitted from the ablation-table notation for clarity.

## B Construction of Binaural AVS-Synthetic

## B.1 Pseudo-Instance Annotation

Binaural AVS-Synthetic is derived from AVS-Synthetic as used by SAMA-AVS (Liu et al. 2024a). The original semantic mask covers all visible regions of the annotated sound category. We apply 8-connected component analysis to this mask and treat the valid connected components as same-class instance candidates. A component is retained only if its area is at least 100 pixels and at least 0.5% of the total semantic foreground. Samples with no valid component or more than five valid components are removed. A component is then sampled with random seed 42 as the sounding instance. The selected component must contain at least 100 pixels and have a bounding-box width and height of at least five pixels. The selected mask is resized to the RGB-image resolution with nearest-neighbor interpolation; its box and centroid are recorded as the sounding-instance annotation.

Connected components are only an initial proposal mechanism, so we perform an additional category-consistency check with GroundingDINO (Liu et al. 2024b). The normalized category name is used as the text query, with both box and text thresholds set to 0.25. We retain a sample only when the number of GroundingDINO detections equals the number of valid mask components. This filtering reduces the 55,705 samples that pass component-level quality control to 31,045 verified samples. The final set contains 4,488 multi-same-class-candidate samples (14.5%).

<table><tr><td>Module</td><td>Dimensions</td><td>Configuration</td></tr><tr><td>Spatial convolution 1</td><td> $2  3 2$ </td><td>Kernel  $1 \times 3 1 ;$  stride  $1 \times 1 0 ;$  padding  $0 \times 1 5 ;$  IN; ReLU</td></tr><tr><td>Spatial convolution 2</td><td> $3 2  6 4$ </td><td>Kernel  $3 \times 1 1$  ; stride  $1 \times 5 ;$  padding  ${ 1 \times 5 } ;$  IN; ReLU</td></tr><tr><td>Spatial convolution 3</td><td> $6 4  1 2 8$ </td><td>Kernel  $3 \times 7 ;$  stride  $2 \times 4 ;$  padding  $1 \times 3 ;$  IN; ReLU</td></tr><tr><td>Spatial query projection</td><td> $1 2 8  2 5 6$ </td><td>Adaptive  $1 \times 1$  pooling followed by a linear projection</td></tr><tr><td>Dense prompt mapper</td><td> $1  6 4  2 5 6$ </td><td>Two  $3 \times 3$  convolutions with GELU between them</td></tr><tr><td>Candidate-region auxiliary head</td><td> $2 5 6  6 4  1$ </td><td> $3 \times 3$  convolution, ReLU, and  $1 \times 1$  convolution</td></tr><tr><td>SAM mask decoder</td><td>256-D inputs</td><td>Two-way Transformer: depth 2, 8 heads, MLP dimension 2048</td></tr></table>

Table A1: Implementation specifications of the key BiSID-AVS modules.

## B.2 Spatially Consistent Binaural Rendering

The dry audio is bound by the pair of image identifier and sound category, which prevents an image containing multiple annotated categories from receiving audio from the wrong category. Depth Anything (Yang et al. 2024a) estimates a relative depth map for each image. We normalize the depth map within each image and sample the value at the selected instance centroid.

Let $\left( c _ { x } , c _ { y } \right)$ be the selected centroid in an image of width W and height H. Its normalized image coordinates are

$$
\begin{array} { c } { { x _ { n } = \displaystyle \mathrm { c l i p } \bigg ( \frac { c _ { x } - W / 2 } { W / 2 } , - 1 , 1 \bigg ) , } } \\ { { y _ { n } = \displaystyle \mathrm { c l i p } \bigg ( \frac { H / 2 - c _ { y } } { H / 2 } , - 1 , 1 \bigg ) . } } \end{array}\tag{A1}
$$

We map these coordinates to azimuth and elevation as

$$
\theta = 7 0 ^ { \circ } x _ { n } , \qquad \phi = 4 0 ^ { \circ } y _ { n } .\tag{A2}
$$

The normalized depth is mapped to a nominal distance in [0.8, 2.2] m. We use the nearest measured direction in the SADIE D1 HRTF database (Armstrong et al. 2018; AudioLab, University of York and Kearney 2024) and convolve the dry waveform with the corresponding left and right head-related impulse responses. Each binaural waveform is cropped or zero-padded to 2 s and peak-normalized to 0.95. Finally, the same 16-kHz, 66-band cochleagram extraction described in the implementation section is applied to both channels.

## B.3 Annotations and Cross-Validation

Each sample contains an RGB image, synchronized binaural waveform, stacked cochleagram, sounding-instance mask and box, and a same-class candidate mask. The candidate mask is used only as auxiliary supervision and for computing the same-class evaluation metrics. The five-fold protocol groups records originating from the same original sample and stratifies the groups by sound category and single-/multicandidate status. For every fold, 10% of the remaining training groups are reserved for validation; each metric is computed separately within each held-out fold on its applicable evaluation subset (single-candidate, multi-candidate, or ful set). The reported result is the unweighted mean of the five fold-specific scores.

## C Construction of Habitat-AVS

## C.1 Scenes, Viewpoints, and Assets

Habitat-AVS is generated with Habitat-Sim, Replica, and the SoundSpaces audio sensor (Savva et al. 2019; Straub et al. 2019; Chen et al. 2020, 2022). It contains four Replica scenes (room\_0, room\_1, office\_0, and office\_2), ten predefined robotic viewpoints, and 14 object categories. The viewpoint identifiers are room0\_0–2, room1\_0–1, office0\_0–1, and office2\_0–2. Each viewpoint specifies a camera position, a look-at point, and a three-dimensional placement range inside the camera field of view. Camera positions are perturbed by at most 0.2 m horizontally and 0.05 m vertically to increase viewpoint variation while preserving the viewpoint identity used for data splitting.

Single-candidate samples contain one target-class object. Multi-candidate samples contain two objects of the same class, of which the first is assigned as the sounding source and the second remains silent. The objects are placed on surfaces found by downward ray casting. Five support rays—one at the center and four around the footprint—must all intersect the scene, and the maximum diference among their hit heights must not exceed 0.05 m.

## C.2 Visibility Control and Ground Truth

Every generated observation passes the following deterministic checks before being saved:

• each target instance contains at least 100 visible pixels;

• its visible mask remains more than five pixels away from every image boundary;

• the visible-mask fill ratio within its 2D bounding box is at least 0.40; and

• the 2D bounding boxes of all inserted objects do not overlap.

If any check fails, object placement is resampled, with at most 50 attempts per sample. Habitat’s semantic identifiers directly provide the pixel-level ground truth. The soundinginstance mask contains only the source identifier, whereas the same-class candidate mask is the union of all target-class identifiers. The sounding-instance bounding box is computed from its visible pixels.

## C.3 Binaural Audio and Evaluation Splits

The audio source is placed at the three-dimensional location of the sounding object. SoundSpaces renders a two-channel room impulse response at 44.1 kHz. For this benchmark, indirect reflections and material-dependent acoustics are disabled so that the generated cue is dominated by source– receiver geometry. The category-matched dry waveform is convolved separately with the left and right impulse responses and peak-normalized. The rendering routine writes a 1-s binaural clip; before feature extraction, it is zero-padded to 2 s and processed into the semantic and spatial audio inputs described above.

The resulting dataset contains 11,781 samples from 14 categories; 45.3% are multi-candidate samples. To prevent nearly identical observations from the same robotic viewpoint from appearing in both training and test data, the ten viewpoints are organized into five fixed pairs. Each fold holds out one pair for testing, and 10% of the remaining viewpointgrouped data is used for validation. For each fold, a foldspecific annotation file assigns every sample to the training, validation, or test split. As with Binaural AVS-Synthetic, each metric is computed separately within each held-out fold on its applicable evaluation subset, and the reported result is the unweighted mean of the five fold-specific scores.

For scene-disjoint evaluation, we additionally construct four leave-one-scene-out folds using room\_0, room\_1, office\_0, and office\_2. In each fold, all viewpoints and samples from one scene are held out for testing, while the other three scenes form the training pool. Ten percent of this pool is reserved for validation, and the checkpoint with the best validation mIoU $\mathrm { \dot { A l l } }$ is evaluated on the held-out scene. Thus, no viewpoint from the test scene appears in training or validation.

## D Baseline Adaptation and Training Protocols

## D.1 Unified Task Adaptation

TPAVI (Zhou et al. 2022), AVSegFormer (Gao et al. 2024), COMBO (Yang et al. 2024b), VCT (Huang et al. 2025), SAMA-AVS (Liu et al. 2024a), and AVISM (Guo et al. 2025) are retrained separately on Binaural AVS-Synthetic and Habitat-AVS using the same five-fold partitions as BiSID-AVS. Their final prediction target is always the true sounding-instance mask, rather than the candidate mask covering all same-class candidates. Monaural methods receive the channel average and retain their oficial audio preprocessing. Methods requiring a video receive a static clip formed by repeating the same RGB image to the clip length expected by their public implementation. The same instance mask is copied to every frame, and no frame-dependent crop, geometric transform, or appearance perturbation is applied. Consequently, the adaptation satisfies the original interfaces without adding motion or cross-frame visual evidence.

We retain the public architecture and recommended optimization strategy of each baseline, except for the input/output changes required by static RGB, binaural-to-monaural conversion, and instance-mask supervision. Model selection is based only on the validation partition of the corresponding fold. All methods use the same sigmoid threshold of 0.5 and are evaluated by the mIoU, mSIL $. \mathrm { A } , F _ { \mathrm { s a m e } } .$ , and SC-DS implementations used for BiSID-AVS.

## D.2 AVISM Adaptation

AVISM (Guo et al. 2025) is designed for video-level audiovisual instance segmentation. For each image, we construct a static clip by repeating it to the native clip length $T$ of the public AVISM configuration. Its audio input is the binauralchannel average, and the sounding-instance mask is repeated over all T frames during training. Following the AVISM evaluation interface, the true sound category and the number of sounding instances $K = 1$ are supplied, but no true mask, target box, or target identity is provided at inference. We select the predicted trajectory with the highest sounding confidence and evaluate its mask on the reference frame. This result therefore measures AVISM under the single-observation setting studied in the paper; it should not be interpreted as an upper bound on AVISM in its native dynamic-video setting.

## D.3 Decision-Level Fusion Baselines

The decision-level baselines use the frozen SAM ViT-H automatic mask generator with its default configuration to obtain category-agnostic visual proposals. The BiSID-AVS spatial response map is upsampled to the image resolution and treated as a sound source localization heatmap (SSL-HM). For a proposal mask $V _ { j }$ and response map $R ,$ the proposal score is

$$
s _ { j } ( R ) = \frac { \sum _ { ( u , v ) } V _ { j } ( u , v ) R ( u , v ) } { \sum _ { ( u , v ) } V _ { j } ( u , v ) + \epsilon } .\tag{A3}
$$

Here, ϵ prevents division by zero. $\mathrm { S S L \mathrm { - } H M + V P }$ uses $R =$ $\hat { S } ^ { \mathrm { s p l } }$ and outputs the proposal with the largest score. Cand. $\mathrm { M a p + S S L \mathrm { - } H M + \Delta V P }$ first weights the response by the predicted candidate-region probability map, $R = \hat { S } ^ { \mathrm { s p l } } \odot$ $\hat { M } ^ { \mathrm { c a n d } }$ , before applying the same proposal-selection rule. The oracle variant replaces the predicted candidate map with the ground-truth candidate mask, $R = \hat { S } ^ { \mathrm { s p l } } \odot M _ { \mathrm { g t } } ^ { \mathrm { c a n d } }$ . All three variants therefore introduce spatial evidence only at the output stage and do not condition visual feature formation or mask decoding.

## D.4 Interaural Counterfactual Evaluation

The interaural counterfactual evaluation is performed on the Habitat-AVS multi-candidate test samples without retraining the model. For the channel-swapping condition, the left and right waveforms of each sample are exchanged while its RGB image remains unchanged. For the mismatched-audio condition, each test image is paired with one binaural waveform randomly drawn from another sample of the same scene and sound category but a diferent source location. Eligible donors are sampled once using random seed 42, yielding one fixed mismatched waveform per test sample. This construction preserves sound-category information while disrupting the correspondence between the binaural spatial cues and the visual source location.